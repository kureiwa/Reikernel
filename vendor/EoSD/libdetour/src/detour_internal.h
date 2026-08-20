#ifndef DETOUR_INTERNAL_H
#define DETOUR_INTERNAL_H

/* Internal header. Not installed; not included by callers. */

#include "detour.h"

#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>   /* ssize_t, ElfW */

/* The 14-byte absolute jump patch: FF 25 00 00 00 00 + 8-byte literal
 * address. Clobbers no registers (varargs-safe). Reaches any 64-bit
 * address. */
#define DETOUR_PATCH_SIZE 14

/* 47-bit user VA on x86_64 with 4-level paging (the Linux default).
 * Used by detour_create's trampoline-placement retry to guard +/-1GB
 * hint arithmetic against overflow into non-canonical addresses, where
 * the kernel would silently substitute its own (possibly far-away)
 * placement. */
#define DETOUR_USER_VA_MAX 0x7FFFFFFFFFFFUL

/* Trampoline buffer size. Relocated prologue (up to ~50 bytes) plus the
 * 14-byte back-jump plus slack. mmap rounds up to one page anyway; this
 * is the usable content bound. */
#define DETOUR_TRAMP_SIZE 64

/* Upper bound on instructions in a 14-byte prologue. Each instruction is
 * at least 1 byte, so 14 is the theoretical max; allow 16 for safety. */
#define DETOUR_MAX_INSNS  16

struct detour {
    void   *target_fn;        /* address of the hooked function           */
    void   *hook_fn;          /* the replacement function                 */
    void  **original_fn_ptr;  /* caller's slot; receives trampoline addr  */
    void   *trampoline;       /* mmap'd buffer with relocated prologue +  */
                              /* 14-byte back-jump to target_fn+patched   */
    uint8_t original_bytes[DETOUR_PATCH_SIZE];  /* saved for disable      */
    size_t  patched_size;     /* total decoded instruction length; may be */
                              /* > 14 if the last instruction spans       */
    int     enabled;
};

/* Decoded instruction. */
typedef struct {
    int     length;           /* total instruction length in bytes        */
    int     is_rip_relative;  /* 1 if a ModR/M byte selects [rip+disp32] */
    int     disp_offset;      /* offset of the 4-byte disp within the     */
                              /* instruction (relative to insn start)     */
    int32_t disp_value;       /* the displacement as read                 */
} detour_insn_t;

/* detour_decode.c -- decode one instruction starting at p. Returns 0 on
 * success, -1 on unsupported opcode or truncated input. `available` is
 * the number of readable bytes starting at p. */
int detour_decode_one(const uint8_t *p, size_t available,
                      detour_insn_t *out);

/* Decode a prologue: repeatedly call detour_decode_one until at least
 * min_bytes are covered. Returns the total decoded length (>= min_bytes)
 * on success, -1 on unsupported opcode or truncated input, 0 if the
 * input was exhausted before reaching min_bytes (function too short). */
ssize_t detour_decode_prologue(const uint8_t *p, size_t available,
                               size_t min_bytes, detour_insn_t *insns,
                               int max_insns, int *out_count);

/* detour_patch.c -- allocate a trampoline buffer. detour_alloc_trampoline
 * tries MAP_32BIT first (keeps the trampoline within +/-2GB of low-memory
 * targets for any future rel32 path), falls back to no hint. Returns NULL
 * on failure.
 *
 * detour_alloc_trampoline_near allocates with `hint` as an advisory
 * address (no MAP_FIXED). Used by detour_create's retry path when
 * MAP_32BIT placed the trampoline too far from a high-address (PIE)
 * target's RIP-relative data; passing target_fn +/-1GB lands the
 * trampoline within the +/-2GB window required for RIP-relative
 * displacement encoding. */
void *detour_alloc_trampoline(void);
void *detour_alloc_trampoline_near(void *hint);
void  detour_free_trampoline(void *trampoline);

/* mprotect the page(s) containing [addr, addr+len) to R+W+X (writable)
 * or R+X (executable). Returns 0 on success, DETOUR_ERR_PROTECT_FAILED
 * on mprotect failure. */
int detour_make_writable(void *addr, size_t len);
int detour_make_executable(void *addr, size_t len);

/* int3-brokered patch of len bytes at target. new_bytes is the full
 * replacement (len bytes). Steps: write 0xCC to byte 0, sync, write
 * bytes 1..len-1, atomically write new_bytes[0], sync, clear_cache.
 * Returns DETOUR_OK. Caller must have made the page writable first. */
int detour_patch(void *target, const uint8_t *new_bytes, size_t len);

/* Issue membarrier(MEMBARRIER_CMD_GLOBAL) to IPI all cores running
 * threads of this process. Falls back to sched_yield() if membarrier
 * returns EINVAL (e.g. kernel without membarrier support). */
void detour_sync_cores(void);

/* Walk dl_iterate_phdr to find the ELF module containing addr, then
 * walk its .dynsym (in-memory via PT_DYNAMIC) for a symbol whose
 * st_value + dlpi_addr == addr. Returns st_size if found, 0 otherwise.
 * When 0 is returned, the caller skips the size check (caller assumes
 * the risk per DESIGN.md). */
size_t detour_lookup_fn_size(void *addr);

#endif /* DETOUR_INTERNAL_H */
