#ifndef DETOUR_H
#define DETOUR_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle. One per hooked function. Allocated by detour_create,
 * freed by detour_destroy. */
typedef struct detour detour_t;

typedef enum {
    DETOUR_OK                  =  0,
    DETOUR_ERR_INVALID         = -1,   /* NULL argument or relocation overflow */
    DETOUR_ERR_ALREADY_HOOKED  = -2,   /* target_fn already has an active hook */
    DETOUR_ERR_UNSUPPORTED_INSN = -3,  /* prologue contains an opcode the
                                        * decoder cannot safely relocate */
    DETOUR_ERR_PROTECT_FAILED  = -4,   /* mprotect(R+W+X) denied (W^X, PaX,
                                        * SELinux execmem) */
    DETOUR_ERR_FN_TOO_SHORT    = -5,   /* symbol size < 14; patch would cross
                                        * into the next function in .text */
} detour_err_t;

/* Prepare a hook on target_fn. Does NOT patch yet; call detour_enable to
 * install the jmp.
 *
 *   target_fn       address of the function to intercept; must already be
 *                   loaded (caller's responsibility; no dlopen hooking).
 *   hook_fn         the replacement function.
 *   original_fn_ptr caller-provided pointer that, after detour_enable,
 *                   receives the address of a trampoline. Calling through
 *                   *original_fn_ptr runs the relocated prologue and jumps
 *                   back into target_fn past the patched bytes, i.e. the
 *                   "call through to original" pattern.
 *   out_handle      receives the opaque handle on success.
 *
 * After detour_enable, *target_fn is the patched bytes (the hook entry).
 * Any caller, whether they cached the function pointer before or after
 * install, goes through the hook. The only way to reach the original code
 * is via *original_fn_ptr.
 *
 * Return: DETOUR_OK on success; DETOUR_ERR_INVALID if any argument is NULL
 * or a RIP-relative displacement cannot be encoded in 32 bits after
 * relocation; DETOUR_ERR_ALREADY_HOOKED if target_fn is already in the
 * hook list; DETOUR_ERR_UNSUPPORTED_INSN if the prologue contains an
 * opcode outside the decoder's supported set; DETOUR_ERR_FN_TOO_SHORT if
 * the ELF symbol for target_fn is shorter than 14 bytes. If no ELF symbol
 * is found for target_fn, the size check is skipped (caller assumes the
 * risk).
 *
 * Thread-safety: not safe to call concurrently with itself or with
 * detour_enable/disable/destroy on the same target_fn without external
 * coordination. The internal hook list is not locked. */
int detour_create(void *target_fn, void *hook_fn, void **original_fn_ptr,
                  detour_t **out_handle);

/* Install the 14-byte absolute jump (FF 25 00 00 00 00 + 8-byte address)
 * into target_fn's prologue. Uses the int3-brokered sequence: write 0xCC
 * over byte 0, synchronize via membarrier(MEMBARRIER_CMD_GLOBAL), write
 * the remaining 13 bytes, atomically replace 0xCC with 0xFF, synchronize
 * again. The .text page is flipped to R+W+X for the patch step and back
 * to R+X afterwards.
 *
 * Thread-safety: safe against other threads calling target_fn during the
 * patch ONLY if a SIGTRAP handler emulates the original first byte. v0.2
 * does not install such a handler; a thread executing in target_fn during
 * the patch window receives SIGTRAP. Other threads not executing in
 * target_fn are unaffected. */
int detour_enable(detour_t *handle);

/* Reverse of detour_enable: restores the original 14 bytes via the same
 * int3-brokered sequence. After return, calls to target_fn go to the
 * original code.
 *
 * Thread-safety: same caveat as detour_enable. */
int detour_disable(detour_t *handle);

/* Disables the hook (if enabled) and frees the handle and its trampoline.
 * Safe to call with NULL handle (no-op). The caller's original_fn_ptr is
 * not touched; the caller must not dereference it after detour_destroy.
 *
 * If the hook is enabled and detour_disable fails (e.g. mprotect denied
 * on a hardened system: W^X, PaX MPROTECT, SELinux execmem),
 * detour_destroy cannot safely free the handle or trampoline: the
 * 14-byte patch is still installed in target_fn's prologue, so the next
 * call to target_fn jumps to hook_fn which dereferences the trampoline
 * through *original_fn_ptr. In this case detour_destroy leaks both and
 * leaves the hook in the internal list (a subsequent detour_create on
 * the same target returns DETOUR_ERR_ALREADY_HOOKED rather than
 * double-patching). The caller may retry detour_destroy after the
 * underlying mprotect constraint is resolved.
 *
 * Thread-safety: not safe to call concurrently with any other operation
 * on the same handle. */
void detour_destroy(detour_t *handle);

#ifdef __cplusplus
}
#endif

#endif /* DETOUR_H */
