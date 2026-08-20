# libdetour: API Design (v0.3 / shipped)

Status: implemented; see `include/detour.h` for the canonical signature
and `DESIGN.md` for the design rationale. All functions are
non-reentrant with respect to the same `target_fn` and require external
coordination for concurrent installation/removal.

## Overview

Inline-hook function interception without `LD_PRELOAD`, by overwriting the
target function's prologue with a jump. Full RIP-relative instruction
detection and trampoline fixup from day one (per original design ambition).
Handle-based so hooks can be toggled on/off without reinstalling. Hooks are
tracked in a dynamically-growing internal list; there's no artificial cap on
how many can be active at once.

## Types

```c
typedef struct detour detour_t;   // opaque handle, one per hooked function

typedef enum {
    DETOUR_OK               = 0,
    DETOUR_ERR_INVALID      = -1,
    DETOUR_ERR_ALREADY_HOOKED = -2,
    DETOUR_ERR_UNSUPPORTED_INSN = -3,  // couldn't safely relocate prologue
    DETOUR_ERR_PROTECT_FAILED = -4,    // mprotect/VirtualProtect failed (W^X, SELinux execmem, PaX)
    DETOUR_ERR_FN_TOO_SHORT = -5,      // (v0.2) target function < 14 bytes; patch would cross
                                      // function boundary
} detour_err_t;
```

## API

```c
// Creates (but does not yet install) a hook: target_fn will jump to hook_fn.
// original_fn_ptr, once enabled, points to a trampoline that calls the
// original (relocated) prologue instructions followed by a jump back into
// target_fn past the overwritten bytes: the normal "call through to
// original" pattern.
//
// target_fn must be loaded before this call (caller's responsibility).
// Hooking functions in libraries dlopen'd after detour_create is out of
// scope; callers must ensure load order.
//
// After detour_enable succeeds, &(*target_fn) returns the address of the
// patched bytes -- i.e. the hook entry. Any caller, whether they cached
// the function pointer before or after install, goes through the hook.
// The only way to reach the original code is via *original_fn_ptr.
//
// Trampoline placement: detour_alloc_trampoline tries MAP_32BIT first
// (low 2GB, which works for non-PIE targets whose RIP-relative data is
// also in the low 2GB). If build_trampoline fails with DETOUR_ERR_INVALID
// because the relocated RIP-relative displacement does not fit in 32 bits
// at the MAP_32BIT address (PIE target whose data is at ~0x5555...),
// detour_create retries in this order:
//   (a) unrestricted mmap(NULL, ...) -- kernel picks the address
//       (lands within +/-2GB of the target's data on some kernels);
//   (b) advisory hint at target_fn + 1GB;
//   (c) advisory hint at target_fn - 1GB.
// Hints (b)/(c) bias the kernel's top-down mmap into the +/-2GB window
// required for RIP-relative encoding. MAP_FIXED is NOT used; each retry
// revalidates the displacement via build_trampoline. NULL hints near the
// user VA boundary (47-bit, 0x7FFFFFFFFFFF) are skipped. Works for both
// non-PIE and PIE binaries. Returns DETOUR_ERR_INVALID if no placement
// can satisfy the +/-2GB RIP-relative displacement constraint.
int detour_create(void *target_fn, void *hook_fn, void **original_fn_ptr,
                   detour_t **out_handle);

// Writes the jmp into target_fn's prologue. Handles RIP-relative addressing
// in the overwritten instructions by relocating them into the trampoline
// with corrected offsets. Uses the int3-brokered patch sequence (write
// 0xCC, membarrier IPI, write remaining 13 bytes, atomic write of the
// new first byte, membarrier IPI). NO SIGTRAP handler is installed: a
// thread executing in target_fn's prologue during the patch window
// receives SIGTRAP. Threads not executing in target_fn are unaffected.
int detour_enable(detour_t *handle);

// Restores the original bytes, hook_fn no longer called.
int detour_disable(detour_t *handle);

// Disables the hook (if enabled) and frees the handle and its trampoline.
// Safe to call with NULL (no-op). The caller's original_fn_ptr is not
// touched; the caller must not dereference it after detour_destroy.
//
// If the hook is enabled and detour_disable fails (mprotect denied on a
// hardened system: W^X, PaX MPROTECT, SELinux execmem), detour_destroy
// cannot safely free the handle or trampoline: the 14-byte patch is
// still installed in target_fn's prologue, so the next call to target_fn
// jumps to hook_fn which dereferences the trampoline through
// *original_fn_ptr. In this case detour_destroy LEAKS the handle, the
// trampoline, and the hook-list entry (so a subsequent detour_create on
// the same target returns DETOUR_ERR_ALREADY_HOOKED rather than
// double-patching). The caller may retry detour_destroy after the
// underlying mprotect constraint is resolved; until then the hook
// remains live.
void detour_destroy(detour_t *handle);
```

## Decoder opcode set

`detour_create` decodes the first 14+ bytes of `target_fn`'s prologue and
relocates the covered instructions into the trampoline. The decoder is a
minimal length-form disassembler: it knows the instructions that appear in
real-world Linux/x86_64 function prologues and rejects anything outside
that set with `DETOUR_ERR_UNSUPPORTED_INSN` rather than guess.

Canonical prologue instructions handled:
- `endbr64` (`F3 0F 1E FA`, 4 bytes; emitted by gcc under
  `-fcf-protection=full` -- the default on recent Ubuntu/Fedora/Arch).
- `sub rsp, imm8` / `sub rsp, imm32` (stack prologue; `Group 1 /5`,
  opcodes `83 /5 ib` and `81 /5 id`).
- `mov rbp, rsp` (frame setup; `MOV r/m,r`, opcode `89` form).
- `push reg` / `pop reg` (`50-5F`, 1 byte).
- single-byte `nop` (`90`, 1 byte; `66 90` is the canonical 2-byte nop,
  with the `0x66` handled as a legacy prefix).
- multi-byte `nop` (`0F 1F /0`, 5-9 bytes; gcc alignment padding before
  function entry).

Wider set also accepted (covers non-prologue bytes the 14-byte patch may
span): legacy prefixes (`0x66`, `0x67`, `LOCK` `0xF0`, `REP` `0xF2`/`0xF3`,
segment overrides `0x2E`/`0x36`/`0x3E`/`0x26`/`0x64`/`0x65`), REX
(`0x40-0x4F`), ALU r/m,r and r,r/m (`00-3B`), `TEST` (`84/85/A8/A9`),
`XCHG` (`86/87`), `MOV` r/m,r and r,r/m (`88-8B`), `MOV` with moffs
(`A0-A3`), `MOV r,imm` (`B0-BF`; imm64 with REX.W, imm16 with `0x66`,
else imm32), `LEA` (`8D`), `JMP`/`CALL` rel (`E8`/`E9`/`EB`), `Group 1`
(`80/81/82/83` r/m,imm), `Group 11` (`C6/C7` r/m,imm -- MOV only),
`Group 3` (`F6/F7` r/m -- TEST /0,/1 take an immediate, /2-/7 do not),
2-byte `0F 1F /0` (NOP), `0F AF` (IMUL r,r/m), `0F B6/B7/BE/BF`
(MOVZX/MOVSX r,r/m).

**RIP-relative detection.** Any ModR/M byte with `mod=00, rm=101` and no
`0x67` address-size prefix selects `[rip+disp32]`. The decoder records the
4-byte displacement's offset within the instruction and its value, so the
relocator in `detour.c` can recompute the displacement for the trampoline's
new address (`new_disp = orig_target - (trampoline + off + length)`). If
`new_disp` does not fit in `int32_t`, `build_trampoline` returns
`DETOUR_ERR_INVALID` and `detour_create` retries with a closer trampoline
placement (see above). A `0x67` prefix turns the same ModR/M form into an
absolute 32-bit displacement (no RIP-relative fixup applied).

## Non-goals

- No support for hooking functions that lack enough safely-relocatable
  prologue bytes for the 14-byte absolute jump patch (see DESIGN.md); these
  return `DETOUR_ERR_UNSUPPORTED_INSN`.
- No support for hooking functions shorter than 14 bytes (the patch would
  cross into the next function in `.text`); `detour_create` uses
  `dl_iterate_phdr` + the in-memory `.dynsym` (only; `.symtab` is not
  consulted) to detect this and returns `DETOUR_ERR_FN_TOO_SHORT`
  (v0.2 addition). Tail-call thunks (`ret`-only), `endbr64; ret` CET stubs,
  and PLT stubs fall in this category.
- No automatic hooking of every call site (this is prologue patching only,
  not a full binary rewriter).
- No PLT/GOT hooking (rewriting the GOT entry rather than the function
  body). PLT/GOT hooking is a separate technique with different tradeoffs
  (catches only PLT-routed calls, doesn't modify `.text`, trivially
  reversible) and is explicitly out of scope.
- No `dlopen` hooking: the caller must ensure the target library is loaded
  before `detour_create`. Hooking `dlopen` itself to install hooks on
  libraries loaded later is out of scope.
- Hot-patch cross-thread safety is partial: the `int3`-brokered write
  sequence is used (single `membarrier(MEMBARRIER_CMD_GLOBAL)` +
  `sched_yield()` fallback, no generation counter), but **no `SIGTRAP`
  handler is installed**. A thread executing in `target_fn`'s prologue
  during the patch window receives `SIGTRAP`. Callers do not need to
  quiesce other threads that are not currently executing in `target_fn`.
  See DESIGN.md "Hot-patch safety".
- Recursive calls re-enter the hook on every call (the patched bytes are
  at function entry, so a self-call goes through the patch again). Hooks
  with non-reentrant state must guard against this.
- The hook list is global and not locked; concurrent
  `detour_create`/`enable`/`disable`/`destroy` requires external
  coordination. Concurrent *callers* of an already-installed hook are
  safe.
