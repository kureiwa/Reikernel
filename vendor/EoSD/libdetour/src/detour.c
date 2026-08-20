/* libdetour v0.2: inline-hook function interception.
 *
 * Patches the first DETOUR_PATCH_SIZE (14) bytes of a target function
 * with FF 25 00 00 00 00 + 8-byte absolute address (jmp [rip+0]). The
 * displaced prologue instructions are decoded, relocated into a
 * trampoline (with RIP-relative displacements fixed up), and the
 * trampoline ends with a 14-byte absolute jump back into the target
 * function past the patched bytes.
 *
 * See API.md and DESIGN.md for the full design rationale.
 */

#include "detour_internal.h"

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Hook list: realloc-based, no artificial cap. */

static detour_t **g_hooks = NULL;
static size_t     g_hook_count = 0;
static size_t     g_hook_cap = 0;

/* Not thread-safe. detour_create/enable/disable/destroy must not be
 * called concurrently from multiple threads without external locking.
 * The int3-brokered patch is cross-thread safe for the patched code's
 * execution, but the hook list management is not. */

static int hook_list_add(detour_t *h)
{
    if (g_hook_count == g_hook_cap) {
        size_t new_cap = g_hook_cap ? g_hook_cap * 2 : 8;
        detour_t **n = realloc(g_hooks, new_cap * sizeof(*n));
        if (!n) return -1;
        g_hooks = n;
        g_hook_cap = new_cap;
    }
    g_hooks[g_hook_count++] = h;
    return 0;
}

static void hook_list_remove(detour_t *h)
{
    for (size_t i = 0; i < g_hook_count; i++) {
        if (g_hooks[i] == h) {
            /* Swap with last and shrink. Order is not preserved. */
            g_hooks[i] = g_hooks[--g_hook_count];
            return;
        }
    }
}

static detour_t *hook_list_find(void *target_fn)
{
    for (size_t i = 0; i < g_hook_count; i++) {
        if (g_hooks[i]->target_fn == target_fn) return g_hooks[i];
    }
    return NULL;
}

/* Trampoline construction */

/* Build the trampoline body: relocated prologue + 14-byte absolute
 * jump back to target_fn + patched_size.
 *
 * Returns 0 on success, DETOUR_ERR_INVALID if a RIP-relative
 * displacement cannot be encoded in 32 bits at the trampoline's address
 * (trampoline too far from the original target). */
static int build_trampoline(void *trampoline, void *target_fn,
                            const detour_insn_t *insns, int n_insns,
                            size_t patched_size)
{
    uint8_t *t = (uint8_t *)trampoline;
    uint8_t *src = (uint8_t *)target_fn;
    size_t off = 0;

    for (int i = 0; i < n_insns; i++) {
        const detour_insn_t *insn = &insns[i];
        memcpy(t + off, src + off, (size_t)insn->length);

        if (insn->is_rip_relative) {
            /* Original target of the RIP-relative access:
             *   insn_start + length + disp_value
             * After relocation to the trampoline, the new displacement
             * must point to the same absolute target:
             *   new_disp = original_target - (trampoline + off + length)
 */
            uintptr_t orig_start = (uintptr_t)src + off;
            uintptr_t orig_target = orig_start + (uintptr_t)insn->length +
                                    (uintptr_t)(int64_t)insn->disp_value;
            uintptr_t new_end = (uintptr_t)t + off +
                                (uintptr_t)insn->length;
            int64_t new_disp = (int64_t)orig_target - (int64_t)new_end;

            if (new_disp < INT32_MIN || new_disp > INT32_MAX) {
                return DETOUR_ERR_INVALID;
            }
            int32_t d = (int32_t)new_disp;
            memcpy(t + off + insn->disp_offset, &d, sizeof(d));
        }
        off += (size_t)insn->length;
    }

    /* Append the 14-byte absolute jump back to target_fn + patched_size. */
    uintptr_t back_target = (uintptr_t)target_fn + patched_size;
    t[off + 0] = 0xFF;
    t[off + 1] = 0x25;
    t[off + 2] = 0x00;
    t[off + 3] = 0x00;
    t[off + 4] = 0x00;
    t[off + 5] = 0x00;
    memcpy(t + off + 6, &back_target, sizeof(back_target));

    return DETOUR_OK;
}

/* Build the 14-byte patch that overwrites target_fn: FF 25 00 00 00 00
 * + 8-byte absolute address of hook_fn. */
static void build_patch(uint8_t patch[DETOUR_PATCH_SIZE], void *hook_fn)
{
    patch[0] = 0xFF;
    patch[1] = 0x25;
    patch[2] = 0x00;
    patch[3] = 0x00;
    patch[4] = 0x00;
    patch[5] = 0x00;
    uintptr_t addr = (uintptr_t)hook_fn;
    memcpy(patch + 6, &addr, sizeof(addr));
}

/* Public API */

int detour_create(void *target_fn, void *hook_fn, void **original_fn_ptr,
                  detour_t **out_handle)
{
    if (!target_fn || !hook_fn || !original_fn_ptr || !out_handle)
        return DETOUR_ERR_INVALID;

    if (hook_list_find(target_fn) != NULL)
        return DETOUR_ERR_ALREADY_HOOKED;

    /* Function-boundary check. If the ELF symbol is found and its size
     * is < 14 bytes, the 14-byte patch would corrupt the next function
     * in .text. If no symbol is found, skip the check (caller assumes
     * the risk per DESIGN.md). */
    size_t fn_size = detour_lookup_fn_size(target_fn);
    if (fn_size != 0 && fn_size < DETOUR_PATCH_SIZE)
        return DETOUR_ERR_FN_TOO_SHORT;

    /* Read up to 32 bytes for decoding. The symbol-size check above
     * guards against reading past the end of a short function when the
     * symbol is known; if unknown, we rely on the caller having ensured
     * the target is a real function (not a tail-call thunk). */
    const size_t decode_window = 32;
    detour_insn_t insns[DETOUR_MAX_INSNS];
    int n_insns = 0;
    ssize_t patched = detour_decode_prologue((const uint8_t *)target_fn,
                                             decode_window, DETOUR_PATCH_SIZE,
                                             insns, DETOUR_MAX_INSNS,
                                             &n_insns);
    if (patched < 0)
        return DETOUR_ERR_UNSUPPORTED_INSN;
    if (patched == 0)
        return DETOUR_ERR_FN_TOO_SHORT;

    detour_t *h = malloc(sizeof(*h));
    if (!h) return DETOUR_ERR_INVALID;
    memset(h, 0, sizeof(*h));
    h->target_fn       = target_fn;
    h->hook_fn         = hook_fn;
    h->original_fn_ptr = original_fn_ptr;
    h->patched_size    = (size_t)patched;
    h->enabled         = 0;

    /* Save the original 14 bytes for disable. Only the first 14 bytes
     * are patched; bytes 14..patched_size-1 (if any) are the tail of
     * the last decoded instruction and are left untouched. */
    memcpy(h->original_bytes, target_fn, DETOUR_PATCH_SIZE);

    /* Allocate and build the trampoline.
     *
     * Strategy 1: MAP_32BIT (low 2GB). Works when target_fn and its
     * RIP-relative data are also in the low 2GB (non-PIE binaries).
     *
     * If build_trampoline fails with DETOUR_ERR_INVALID, the trampoline
     * is too far from a high-address (PIE) target's RIP-relative data
     * and the 32-bit displacement cannot be encoded. Retry with
     * progressively closer placements:
     *   (a) unrestricted (no MAP_32BIT, no hint) -- the 14-byte absolute
     *       back-jump in the trampoline reaches any 64-bit address;
     *       this is sufficient when the kernel happens to place the
     *       trampoline within +/-2GB of the target's RIP-relative data;
     *   (b) advisory hint at target_fn + 1GB;
     *   (c) advisory hint at target_fn - 1GB.
     * Hints (b)/(c) land the trampoline within the +/-2GB window
     * required for RIP-relative displacement encoding on kernels whose
     * top-down mmap would otherwise place the unrestricted allocation
     * far from a PIE target. */
    h->trampoline = detour_alloc_trampoline();
    if (!h->trampoline) {
        free(h);
        return DETOUR_ERR_INVALID;
    }

    int rc = build_trampoline(h->trampoline, target_fn, insns, n_insns,
                              h->patched_size);
    if (rc == DETOUR_ERR_INVALID) {
        detour_free_trampoline(h->trampoline);
        h->trampoline = NULL;

        /* Build the retry candidate list. The first entry is an
         * unrestricted allocation (no MAP_32BIT, no hint); the next
         * two are advisory +/-1GB hints around target_fn. NULL hints
         * are skipped (e.g. target_fn near the user VA boundary). */
        void *candidates[3] = { NULL, NULL, NULL };
        int n_cand = 0;
        candidates[n_cand++] = NULL;  /* unrestricted */
        uintptr_t taddr = (uintptr_t)target_fn;
        if (taddr + 0x40000000UL <= DETOUR_USER_VA_MAX)
            candidates[n_cand++] = (void *)(taddr + 0x40000000UL);
        if (taddr >= 0x40000000UL)
            candidates[n_cand++] = (void *)(taddr - 0x40000000UL);

        for (int i = 0; i < n_cand && h->trampoline == NULL; i++) {
            void *p = detour_alloc_trampoline_near(candidates[i]);
            if (p == NULL) continue;
            rc = build_trampoline(p, target_fn, insns, n_insns,
                                  h->patched_size);
            if (rc == DETOUR_OK) {
                h->trampoline = p;
            } else {
                detour_free_trampoline(p);
                rc = DETOUR_ERR_INVALID;  /* keep trying */
            }
        }
    }
    if (rc != DETOUR_OK || h->trampoline == NULL) {
        if (h->trampoline) detour_free_trampoline(h->trampoline);
        free(h);
        return (rc != DETOUR_OK) ? rc : DETOUR_ERR_INVALID;
    }

    /* Publish the trampoline address to the caller. The trampoline is
     * usable immediately: it runs the relocated prologue and jumps to
     * target_fn + patched_size, which is valid code whether or not the
     * hook is enabled. */
    *original_fn_ptr = h->trampoline;

    if (hook_list_add(h) != 0) {
        detour_free_trampoline(h->trampoline);
        free(h);
        return DETOUR_ERR_INVALID;
    }

    *out_handle = h;
    return DETOUR_OK;
}

int detour_enable(detour_t *handle)
{
    if (!handle) return DETOUR_ERR_INVALID;
    if (handle->enabled) return DETOUR_OK;

    int rc = detour_make_writable(handle->target_fn, DETOUR_PATCH_SIZE);
    if (rc != DETOUR_OK) return rc;

    uint8_t patch[DETOUR_PATCH_SIZE];
    build_patch(patch, handle->hook_fn);

    detour_patch(handle->target_fn, patch, DETOUR_PATCH_SIZE);

    /* Restore R+X. The W^X window existed only during the patch step. */
    detour_make_executable(handle->target_fn, DETOUR_PATCH_SIZE);

    handle->enabled = 1;
    return DETOUR_OK;
}

int detour_disable(detour_t *handle)
{
    if (!handle) return DETOUR_ERR_INVALID;
    if (!handle->enabled) return DETOUR_OK;

    int rc = detour_make_writable(handle->target_fn, DETOUR_PATCH_SIZE);
    if (rc != DETOUR_OK) return rc;

    /* Restore the original 14 bytes via the same int3-brokered sequence. */
    detour_patch(handle->target_fn, handle->original_bytes,
                 DETOUR_PATCH_SIZE);

    detour_make_executable(handle->target_fn, DETOUR_PATCH_SIZE);

    handle->enabled = 0;
    return DETOUR_OK;
}

void detour_destroy(detour_t *handle)
{
    if (!handle) return;
    if (handle->enabled) {
        int rc = detour_disable(handle);
        if (rc != DETOUR_OK) {
            /* detour_disable failed (typically mprotect denied on a
             * hardened system: W^X, PaX MPROTECT, SELinux execmem).
             * The 14-byte patch is still installed in target_fn's
             * prologue, and any caller of target_fn jumps to hook_fn,
             * which dereferences *handle->original_fn_ptr (the
             * trampoline). Freeing the handle or trampoline now would
             * create a use-after-free on the next call to target_fn.
             *
             * Leak the handle, the trampoline, and the hook-list entry
             * (so a subsequent detour_create on the same target returns
             * DETOUR_ERR_ALREADY_HOOKED rather than double-patching).
             * The caller may retry detour_destroy after the underlying
             * mprotect constraint is resolved; until then the hook
             * remains live. */
            return;
        }
    }
    hook_list_remove(handle);
    detour_free_trampoline(handle->trampoline);
    free(handle);
}
