# libdetour: Design Notes (v0.3 / shipped)

## Problem

Intercept function calls (malloc/free for leak detection, syscalls for
container policy enforcement, legacy ABI shims) in statically-linked
binaries or scenarios where `LD_PRELOAD` isn't available/sufficient, by
directly patching the target function's machine code prologue.

## Why full RIP-relative fixup from day one

The simpler alternative (only support functions with no RIP-relative
addressing in the first few bytes) would silently fail on a large fraction
of real-world compiled functions, since RIP-relative addressing is common
in position-independent code (the default on modern Linux). Given this is a
learning/mastery-focused project and the original notes already scoped this
correctly ("well-understood technique used by Microsoft Detours, not
experimental"), doing it properly from the start avoids a half-working
first cut that needs a rewrite.

## The ASM/decoding boundary

Two pieces of non-trivial work here, both are the hard core of this
module:

1. **x86-64 instruction length decoding.** To know how many prologue bytes
   are safe to overwrite (must overwrite whole instructions, never split
   one), libdetour needs a minimal length disassembler. **Resolved scope:**
   must correctly decode `push`, `mov` (register and r/m forms),
   immediate-operand instructions, `jmp rel8`/`rel32`, `call rel32`, `test`,
   `cmp`, `xor`, `lea`, `sub rsp, imm32` / `sub rsp, imm8` (canonical stack
   prologue), `mov rbp, rsp` (frame setup), `endbr64` (`F3 0F 1E FA`, emitted
   by gcc under `-fcf-protection=full` -- the default on recent Ubuntu/Fedora/
   Arch), single-byte `nop` (`90`), and multi-byte `nop` (`0F 1F /0` family,
   emitted by gcc for alignment padding before function entry), plus REX
   prefixes and 0x66/0x67 operand/address-size override prefixes. Any opcode
   outside this set encountered while scanning the prologue causes
   `detour_create` to fail with `DETOUR_ERR_UNSUPPORTED_INSN` rather than
   guessing. Expanding the set later (e.g. SSE/AVX prologue instructions)
   is additive, not a breaking change.
2. **RIP-relative relocation.** When a relocated instruction used
   `rip`-relative addressing, the offset must be recalculated for the new
   location in the trampoline (the target address is fixed, but the
   instruction's new address changes the required relative offset).

## Patch size: resolved as 14-byte absolute jump

A 5-byte `jmp rel32` is insufficient in the general case: the hook target
may be more than +/-2GB from the patched function, which a `rel32` can't
reach reliably without also managing trampoline placement within range
(possible but adds real complexity). **Resolved:** always use the 14-byte
absolute jump sequence `FF 25 00 00 00 00` + 8-byte literal address
(`jmp [rip+0]` followed by the absolute target). This is the same form used
by Microsoft Detours; it clobbers no registers (so it is safe for SysV
x86_64 varargs where AL carries the vector-register count) and reaches any
address in the 64-bit space. The originally-named `push rax; mov rax, imm64;
jmp rax` sequence is 13 bytes (not 14) and clobbers RAX, so it is not used.

The trampoline's back-jump (relocated prologue → original function past the
patched bytes) uses the same 14-byte form, so trampolines can be allocated
anywhere in the address space -- *for the back-jump*. The relocated prologue
instructions themselves, however, may use RIP-relative addressing, and a
RIP-relative displacement must fit in 32 bits. The trampoline therefore has
a placement constraint: it must lie within ±2GB of any RIP-relative target
referenced by the relocated prologue (typically a global in the target's
module, very close to the target function itself).

Each `detour_create` allocates a fresh trampoline via `mmap` (one page,
PROT_READ|PROT_WRITE|PROT_EXEC, MAP_PRIVATE|MAP_ANONYMOUS); each
`detour_destroy` releases it via `munmap`. The allocator's strategy is
`MAP_32BIT` first (places the trampoline in the low 2GB, which works when
the target and its RIP-relative data are also in the low 2GB -- i.e.
non-PIE binaries). On relocation overflow (`DETOUR_ERR_INVALID` from
`build_trampoline`, because the relocated RIP-relative displacement does
not fit in 32 bits at the MAP_32BIT address), `detour_create` retries with
progressively closer placements:

  (a) unrestricted `mmap(NULL, ...)` -- kernel picks (works when the
      kernel happens to place the allocation within +/-2GB of the target's
      data);
  (b) advisory hint at `target_fn + 1GB`;
  (c) advisory hint at `target_fn - 1GB`.

Hints (b)/(c) are needed for PIE binaries: the kernel's default top-down
`mmap` lands unrestricted allocations near `mmap_base` (~0x7fff...), which
is more than 2GB from a PIE target's text/data at ~0x5555.... Biasing the
hint to `target_fn +/- 1GB` lands the trampoline within the +/-2GB window
required for RIP-relative encoding. `MAP_FIXED` is not used: the hint is
advisory, and `build_trampoline` revalidates the displacement on each
retry. Hints that would overflow the 47-bit user VA limit
(`0x7FFFFFFFFFFF`) are skipped (the kernel would silently substitute its
own placement, defeating the hint).

This is directly why the instruction decoder above must handle a reasonably
wide instruction set: a 14-byte patch is more likely to span multiple real
instructions than a 5-byte one.

## Function-boundary overflow detection

A 14-byte patch on a function shorter than 14 bytes would corrupt the next
function in `.text`. libdetour uses `dl_iterate_phdr` to walk the ELF
program headers and the in-memory `.dynsym` (via `PT_DYNAMIC` / `DT_HASH`
or `DT_GNU_HASH`) to determine the size of the target function. **Only
`.dynsym` is consulted**; `.symtab` is not read (it is not mapped at runtime
for stripped binaries, and reading it from disk is out of scope). If the
target's symbol size is less than 14 bytes, `detour_create` returns
`DETOUR_ERR_FN_TOO_SHORT`. This catches tail-call thunks (`ret`-only,
1 byte), `endbr64; ret` CET stubs (5 bytes), and PLT stubs (16 bytes but
with internal structure that must not be partially overwritten). When no
symbol is found for `target_fn` (e.g. a static function in a binary built
without `-rdynamic`, or any function in a fully-statically-linked binary
with no `PT_DYNAMIC`), the check is skipped and the caller assumes the
risk. Tests use `-rdynamic` to export their symbols to `.dynsym` so the
size check fires.

## Hot-patch safety: int3-brokered patching

Cross-thread code patching on x86 uses the `int3`-brokered sequence
pioneered by the Linux kernel's `text_poke_bp()` (in `arch/x86/kernel/
alternative.c`) and by ftrace's `ftrace_modify_code`. The install sequence:

1. Atomically write `0xCC` (`int3`, 1 byte) over the first byte of the
   target instruction. Aligned 1-byte stores are atomic on x86.
2. Synchronize: a single `membarrier(MEMBARRIER_CMD_GLOBAL, 0)` call issues
   an IPI to all cores running threads of this process; on x86, interrupt
   delivery is a serializing event for the instruction stream. If
   `membarrier` returns `EINVAL` (kernel without membarrier support), the
   fallback is a single `sched_yield()` -- degraded; it does not
   synchronize other cores, only gives the scheduler a chance to migrate.
   There is no generation counter and no retry loop.
3. Write the remaining 13 patch bytes. Non-atomic `memcpy` is safe because
   any thread fetching byte 0 sees `int3` and traps before reaching bytes
   1..13.
4. Atomically replace `0xCC` with the new first byte (`FF` for the
   `jmp [rip+0]` form).
5. Synchronize again (same `membarrier` + `sched_yield` fallback).

**No `SIGTRAP` handler is installed.** The kernel's `text_poke_bp` routes
an `int3` trap from a thread executing in the patched function into an
emulation of the not-yet-installed prologue; libdetour does not. A thread
executing in `target_fn`'s prologue during the patch window therefore
receives `SIGTRAP` and (without an application-installed handler)
terminates. Threads not executing in `target_fn` are unaffected. This is
the documented limitation (see `detour.h`'s `detour_enable` comment). The
full cross-thread safety of `text_poke_bp` (int3 handler + emulate +
single-step + resume) is a future-work item. Intel SDM Vol 3A §8.1.3
requires this kind of serialization for cross-modifying code.
`__builtin___clear_cache` is called after every patch step on all
platforms -- it is a no-op on x86_64 but marks intent and is required for
the future ARM64 port.

## Why unbounded hook list, not fixed capacity

Unlike `libtick`'s timer registry (predictable, bounded per event loop),
the number of functions someone might want to hook in a debugging/tracing
tool is inherently open-ended and use-case dependent (original notes
mention "10-50" as typical, but a capacity cap would be an arbitrary,
easily-hit limit for no real benefit). A dynamically-growing list (realloc-
based) is used instead.

## W^X hardening

`mprotect(addr, len, PROT_READ | PROT_WRITE | PROT_EXEC)` is required to
make `.text` writable for the patch. On mainline Linux this succeeds. On
PaX/grsec `MPROTECT`, strict SELinux `execmem` policies, and hardened
malloc (scudo, GrapheneOS) it is denied. No fallback is implemented (no
alternate mapping, no `mmap`-a-copy-and-remap trick); on failure,
`detour_enable` returns `DETOUR_ERR_PROTECT_FAILED`. The W^X window is
intentionally narrow: `detour_enable` flips the page(s) containing the
patched range to `R+W+X` for the duration of the int3-brokered write
sequence, then immediately flips back to `R+X` before returning. The same
flip-write-unflip pattern is used by `detour_disable`. The window spans
two `membarrier` IPIs + one non-atomic `memcpy` of 13 bytes + two atomic
1-byte stores + two `mprotect` syscalls -- microseconds-scale. There is no
`R+W+X` page left behind in steady state; the only persistent `R+W+X`
pages in the process are the trampolines (`PROT_READ|PROT_WRITE|PROT_EXEC`
from `mmap`, because the trampoline is both written once at create time
and executed on every hook invocation).

If `detour_disable` fails inside `detour_destroy` (mprotect denied on a
hardened system, the patch still installed), `detour_destroy` does **not**
free the handle or trampoline: doing so would create a use-after-free on
the next call to `target_fn` (which would jump to `hook_fn` → dereference
the freed trampoline through `*original_fn_ptr`). It leaks both and leaves
the hook in the internal list so a subsequent `detour_create` on the same
target returns `DETOUR_ERR_ALREADY_HOOKED` rather than double-patching.
The caller may retry `detour_destroy` after the underlying mprotect
constraint is resolved.

## Benchmarks

Measured on a 2-online-CPU host (gcc 14.2.0, `-O2`, glibc, Linux x86_64).
See `bench/bench_overhead.c` and `bench/bench_toggle.c`.

- **Hook overhead per call: ~4.0 cycles.** Measured as
  `(hooked_call - direct_call)` over 5,000,000 iterations using `rdtscp`.
  Typical: direct call ~6.7 cycles, hooked call ~10.8 cycles, delta ~4.1.
  The overhead is the indirect `jmp [rip+0]` in the patch + the
  trampoline's relocated prologue + the 14-byte absolute back-jump. Run-
  to-run variance is ~0.3 cycles.

- **enable/disable round-trip: ~24 ms per pair.** Measured by
  `bench_toggle` over 1000 iterations using `CLOCK_MONOTONIC` (rdtsc would
  conflate TSC progress with wall-clock time across syscalls + IPIs). Each
  round-trip is 2x `mprotect` + 2x `membarrier(MEMBARRIER_CMD_GLOBAL)`;
  the cost is dominated by the two membarrier IPIs (each round-trips to
  every core running a thread of this process) and the two `mprotect`
  syscalls (each is a kernel entry to flip page protections). Scales
  roughly linearly with online CPU count.

- **enable (already-enabled): ~12 us/op.** API no-op (`detour_enable` on
  an enabled hook returns `DETOUR_OK` without patching) but still pays
  the function-entry overhead.

- **disable (already-disabled): ~2 ns/op.** Fast path: single branch on
  `handle->enabled`.

`bench_toggle` measures the round-trip rather than enable-alone /
disable-alone because the only way to keep the untimed counterpart out of
the timed region would be to batch ITERS enables followed by ITERS
disables, which makes all but the first enable a no-op and does not
measure the patch path. The round-trip is the operation a caller who
installs and removes a hook on every use actually pays.

## Non-goals

- Not a full disassembler or binary analysis tool.
- Cross-thread patch installation safety is partial: the `int3`-brokered
  write sequence is used, but **no `SIGTRAP` handler is installed** to
  emulate the not-yet-installed prologue. A thread executing in
  `target_fn`'s prologue during the patch window receives `SIGTRAP`.
  Threads not executing in `target_fn` are unaffected. Full
  `text_poke_bp`-style safety (int3 handler + emulate + single-step +
  resume) is future work.
- The hook list is global and **not locked**; concurrent
  `detour_create` / `detour_enable` / `detour_disable` / `detour_destroy`
  on the same or different targets races. The int3-brokered patch is safe
  for concurrent *callers* of an already-installed hook, but hook
  installation/removal itself requires external coordination.
- Recursive calls through the patched entry re-enter the hook on every
  call (the patched bytes are at function entry, so a self-call goes
  through the patch again). Hooks with non-reentrant state must guard
  against this.
- No Windows PE-specific trampoline handling (ELF/System V only, per
  toolkit-level platform decision).
- No PLT/GOT hooking (see API.md).
- No `dlopen` hooking: caller must ensure target is loaded before
  `detour_create`.
- CET-IBT enforced environments are not supported: the trampoline's
  back-jump is an indirect `jmp [rip+0]` whose landing site at
  `target_fn + patched_size` typically does not start with `endbr64`.
