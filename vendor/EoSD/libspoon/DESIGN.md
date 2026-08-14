# libspoon: Design Notes (v0.3, shipped)

## Problem

Cooperative context switching cheaper than `swapcontext` (~500ns) or a
kernel thread context switch (~1-2us), targeting ~50-100ns via hand-written
NASM (ballpark, verified by `bench/bench_switch.c`). Use cases: HTTP/2-style
stream multiplexing, embedded GUI screen switching, staged simulation
pipelines, all scenarios where the caller already knows the right switch
order and doesn't need a scheduler making that decision for it.

## Why no built-in scheduler

Every candidate use case in the original notes already has an obvious,
domain-specific switching order (next HTTP stream with pending data, next
GUI screen on a 16ms tick, next simulation stage). A generic round-robin
scheduler would need to be overridden immediately by every real caller, so
libspoon skips it. `spoon_switch_to` and `spoon_yield` are the only two
primitives; callers build their own policy on top.

## Why per-coroutine stack size override

Fixed-size-only was tempting for simplicity, but the original use cases span
wildly different stack needs (a simple GUI screen switch vs. a deep
recursive simulation stage). Pool-wide default keeps the common case
one-line-simple; override handles the outliers without a second API.

The minimum is fixed at `SPOON_MIN_STACK` (16 KB, matching glibc's
`PTHREAD_STACK_MIN`, the safe floor for signal delivery). Pool default
below this is rejected at `spoon_pool_create`; per-coroutine override below
this is rejected at `spoon_create` with `SPOON_ERR_INVALID`. There is no
silent clamping -- a caller supplying a fixed-size `alloc_hook` buffer
would otherwise have libspoon write past the end of that buffer once
the clamp bumped the effective size above what the hook returned.

## The asymmetric invariant

Each coroutine has at most one caller at a time -- the context that
most recently switched to it. The caller chain forms a tree rooted
at the main thread (whose scratch context has `caller == NULL`).

`spoon_switch_to` overwrites `co->caller` to point at the from-context
before the switch. If `co` is already in `from`'s caller chain (the
nested A->B->C->A case, or the self-switch case `co == from`), that
overwrite creates a cycle: when `co` later finishes and switches
back through its caller chain, control eventually re-enters a
context whose saved state has already been consumed on the way down
-- a SIGABRT.

`spoon_switch_to` detects and rejects this before the switch by walking
`from`'s caller chain and returning `SPOON_ERR_INVALID` if `co`
appears anywhere in it. The same walk covers the self-switch case
(`co == from`). The check is O(depth-of-caller-chain), which is bounded
by the pool capacity in the worst case and is 1 in the common
main->co->main pattern. Asymmetric coroutines therefore cannot be
nested symmetrically; callers wanting symmetric handoff must build
it explicitly on top of `spoon_switch_to` and `spoon_yield` (typically
by tracking a "next" handle in user data and switching to that).

## `spoon_yield` as a convenience primitive

`spoon_yield` is exactly `spoon_switch_to(caller)` with the caller
handle read from the current coroutine's `caller` field, set by
whichever `spoon_switch_to` call resumed us. Without it, every
coroutine function would have to receive its caller's handle through
its `arg` parameter (or stash it in a thread-local) and call
`spoon_switch_to` explicitly. Yield exists so the common
"cooperative coroutine that hands control back to its driver" pattern
is one call, not a small dance.

It refuses to do anything outside a coroutine: called from the main
thread (`tl_current == NULL`) or from a coroutine whose `caller` is
NULL (should not happen in normal use, since the only way to enter
a coroutine is via `spoon_switch_to`, which sets `caller` first), it
returns `SPOON_ERR_INVALID` rather than dereferencing a null pointer.
The status is set to `SPOON_SUSPENDED` before the switch so the caller
observes the right state on resume.

## The ASM boundary

`spoon_switch_to` is the only function requiring hand-written NASM. Per the
toolkit-level spec, this needs a System V AMD64 version now, with a Windows
x64 version deferred until cross-platform work begins (different
register/arg conventions: `rdi/rsi` vs `rcx/rdx`, different callee-saved
register set, and Windows requires saving more of the stack/TIB state).
The switch itself is roughly 21 instructions across five logical phases:
save outgoing context (callee-saved GPRs + MXCSR + x87 CW + RSP), load
incoming RSP, restore incoming context, load incoming RIP, transfer
control. XMM/YMM register values are caller-saved per the System V ABI and
are not preserved (and need not be).

### Switch instruction count (23 -> 21)

The original NASM switch was 23 instructions. Two optimizations brought
it to 21:

1. **RIP save via `pop rax`.** The return address lives at `[rsp]`
   (pushed by `call spoon_switch`). The original sequence was
   `mov rax, [rsp]` / `lea rdx, [rsp+8]` / `mov [rdi + OFF_SP], rdx` /
   `mov [rdi + OFF_RIP], rax` -- four instructions, with `lea` computing
   the pre-call RSP. `pop rax` reads the return address into `rax` and
   advances `rsp` past it as a side effect. The side effect is invisible
   because `rsp` is overwritten by `mov rsp, [rsi + OFF_SP]` before any
   further stack access. The `pop` form saves one instruction: `pop rax`
   / `mov [rdi + OFF_RIP], rax` / `mov [rdi + OFF_SP], rsp` is three
   instructions for the same effect.
2. **Indirect `jmp [mem]` instead of `mov rax, [mem]; jmp rax`.** The
   final control transfer loads the incoming RIP from `[rsi + OFF_RIP]`
   and jumps to it. `jmp [rsi + OFF_RIP]` does both in one instruction
   because the memory operand is the jump target. The two-instruction
   `mov` + `jmp rax` form was used originally for symmetry with the
   save side; the indirect form is shorter and equivalent because no
   later instruction needs the loaded RIP in a register.

Neither optimization changes the context-struct layout or the set of
registers preserved. Both are purely instruction-count reductions
inside `spoon_switch`.

### MXCSR + x87 control word preservation

The switch preserves MXCSR control bits and the x87 control word
(both callee-saved per System V ABI table 3.4 + section 3.2.2) via
`stmxcsr` / `ldmxcsr` and `fnstcw` / `fldcw` -- four instructions
(two on save, two on restore), six extra bytes in the context struct
(`mxcsr` u32 + `x87_cw` u16 + 2 bytes padding), negligible cost vs.
the 50-100ns target. This matches what glibc `swapcontext` and
Boost.Context's `jump_fcontext` do. XMM/YMM register values are
caller-saved per the ABI and are not preserved; if a coroutine needs
floating-point/SIMD values to survive a yield, it must save them
itself. `tests/test_fp.c` exercises the MXCSR path: a coroutine sets
round-down mode, yields, and asserts the mode is still round-down on
resume; the main thread asserts its own MXCSR is unchanged across the
yield.

## Why stack overflow protection is not built in

`libsva` already solves guarded memory regions generally (guard pages,
executable memory, etc.). Duplicating that logic inside libspoon would
violate the "fully independent, zero cross-deps" architecture decision at
the toolkit level, and it would also just be redundant. `spoon_create_with_stack()`
accepts `alloc_hook`/`free_hook` function pointers, so a caller can pass
hooks that wrap an already-`sva_map_guarded` region instead of using the
pool's default allocator. Example usage (for the toolkit README / this
module's examples folder, not in library source):

```c
void *my_alloc(size_t size, void *user_data) {
    sva_err_t err;
    sva_region_t *r = sva_map_guarded(size, SVA_PROT_READ | SVA_PROT_WRITE, &err);
    return r ? sva_base(r) : NULL;   // region handle itself would need to be
                                      // tracked separately by user_data for free_hook
}
```

Exact example plumbing (tracking the `sva_region_t*` alongside the raw
pointer for the matching `free_hook` call) is left to the example code
in `EoSD/example/spoon_sva.c`; the API shape above is what's locked in.

## Non-goals

- No scheduler.
- No symmetric coroutine handoff. Asymmetric coroutines cannot be
  nested symmetrically (A->B->C->A is rejected); callers wanting
  symmetric semantics must build them on top of `spoon_switch_to` and
  `spoon_yield`.
- No built-in overflow protection.
- No cross-thread coroutine handoff.
- No XMM/YMM register preservation across switches (caller-saved per ABI;
  caller's responsibility). MXCSR control bits and x87 control word ARE
  preserved (callee-saved per ABI).
