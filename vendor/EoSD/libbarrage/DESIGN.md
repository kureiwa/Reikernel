# libbarrage: Design Notes (v0.3)

## Problem

`malloc` is too slow and fragmentation-prone for short-lived, high-volume
allocation patterns (per-HTTP-request scratch data, per-physics-tick
transient objects, per-script-call marshaling buffers) where everything can
be freed at once when the logical unit of work ends.

## Why fixed-size backing store, no growth

Growable arenas (additional mmap'd chunks when exhausted) add real
complexity: allocations can no longer be a guaranteed-contiguous single
region, which complicates both the bump-pointer fast path (now needs to
check "am I in the last chunk" logic) and any code that assumes arena
memory is one contiguous block. Given the target use cases (per-request,
per-frame) have a roughly predictable upper bound on memory needed, a fixed
ceiling with a clear error on overflow is simpler and keeps the allocation
fast path at the "handful of instructions" level described in the original
notes. Callers who genuinely can't bound their allocation needs should size
generously or use a different allocator. That's an explicit, accepted
v0.3 limitation.

## Why no finalizers on reset

A finalizer list would mean `barrage_alloc` needs to optionally register a
cleanup callback per allocation, turning the "handful of instructions" fast
path into something that has to check and possibly append to a callback
list, directly undermining the entire point of this allocator. The stated
use cases (request-scoped strings/headers, transient collision manifolds,
temporary marshaling structures) are all plain data with no external
resources (file handles, locks, etc.) needing cleanup, so bulk-free-with-
no-callbacks is the right fit. If a real need for finalizers emerges, that's
a different allocator, not a feature bolted onto this one.

## The ASM boundary

The bump-allocate fast path aligns the bump pointer BEFORE advancing by
`size`, so the returned pointer is the aligned base of the new allocation
and the bounds check tests the aligned end. The literal sequence (for a
16-byte alignment, the common case):

```
mov rax, [top]              ; current top
lea rax, [rax+15]           ; round up to 16-byte boundary
and rax, -16                ; rax = align_up(top, 16) -- return value
lea rcx, [rax+size]         ; new top after this allocation
cmp rcx, [end]              ; bounds check the ALIGNED new top
ja  fallback                ; out of space
mov [top], rcx              ; commit
```

For other alignments, the `15` / `-16` constants become `align-1` /
`-align`. The `barrage_arena_t` struct is padded to `alignas(64)` to
prevent false sharing between per-thread arenas allocated from the same
malloc heap. The arena's `top` is kept aligned to 16 bytes as an invariant
after `barrage_create`, so the alignment step in the fast path can be
elided for the default `align=16` case (an optimization the implementer
may take or leave).

The align-before-advance ordering is a v0.2 fix. The earlier sequence
advanced `top` first and aligned second, returning a pointer at
`align_up(top_old + size, align)` rather than at `align_up(top_old,
align)`. That pointer is the start of the *next* allocation's territory,
not the aligned base of the current one: for `align=16` it sits up to 15
bytes past the end of what the caller asked for, so the next allocation's
first bytes overlap the current one's tail, and the bounds check (which
ran against the unaligned `top_old + size`) didn't catch it. Aligning
first means the bounds check sees the aligned end and the returned
pointer is the aligned base of the new allocation, as the caller expects.

## Backing store dispatch

The backing store is one allocation captured at `barrage_create` time and
released as one allocation in `barrage_destroy`. The create-side dispatch
picks the backing store in priority order: caller's `allocator.alloc`
(after pair validation) first, then the default path. The default path
splits on a 1 MiB threshold:

  - `size >= 1 MiB` -> `mmap(NULL, size, PROT_READ | PROT_WRITE,
    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)`. Anonymous private mappings are
    zero-initialized and copy-on-write; the kernel returns a page-aligned
    address, which subsumes every alignment libbarrage promises (max 64).
    Page-table cost is amortized over a large region.
  - `size < 1 MiB` -> libc `malloc`. For small arenas mmap's page-table
    churn dominates and madvise on a sub-page region is a no-op anyway, so
    malloc is cheaper and equally correct (glibc returns
    `alignof(max_align_t)`-aligned memory, 16 bytes on x86_64, which
    satisfies the 64-byte cap via the per-allocation align step).

The destroy side has to remember which default path was taken. It does so
with a `backing_via_mmap` flag set at create time: 1 means munmap on
destroy, 0 means libc free or the user's free. The three-way dispatch in
`barrage_destroy` is: `backing_via_mmap` -> `munmap`; else
`allocator.free` non-NULL -> user's free; else libc `free`. See API.md
"Backing store and destroy dispatch" for the full priority order and the
invariant that keeps the dispatch unambiguous.

## Allocator pair validation

The `barrage_allocator_t` struct has separate `alloc` and `free` fields,
but they must be supplied as a pair: both non-NULL or both NULL. If the
caller passes a struct with exactly one set, `barrage_create` rejects the
whole struct (sets both fields to NULL) and uses the default path. This
is a v0.2 fix for an asymmetric-dispatch bug:

  - The create-side dispatch checked only `allocator.alloc`. If `alloc`
    was NULL and `free` was non-NULL, the default path was used and the
    user's `free` was ignored at create time.
  - The destroy-side dispatch checked `allocator.free` directly. With
    `free` non-NULL, it called the user's `free` on memory that came
    from libc `malloc` (or mmap) -- a type mismatch and a crash waiting
    to happen, since the user's `free` was almost certainly
    `something_other_than_libc_free`.

Rejecting half-supplied allocators at create time makes the dispatch
unambiguous: `allocator.free` is non-NULL only when `allocator.alloc` was
actually used to obtain the backing store, so destroy's fallthrough from
`backing_via_mmap` to `allocator.free` to libc `free` always picks the
right one. `tests/test_edge.c` covers the three cases (pair, alloc-only,
free-only) as regressions.

## Why no internal locking

Matches the module's name and stated purpose (per-thread). Adding a lock
would mean paying synchronization cost on every allocation to protect
against a usage pattern (sharing one arena across threads) that isn't the
intended use anyway. The fix for multi-threaded workloads is "one arena
per thread," the same pattern used by `libtick` and `libspoon`. The
`barrage_arena_t` struct is `alignas(64)` to prevent false sharing between
per-thread arenas.

## Performance

`bench/bench_alloc.c` measures 1M 32-byte allocations with `align=16` in
a 256 MiB arena and divides total elapsed time by N. Result: 2.3-2.8
ns/op for `barrage_alloc` vs ~25-40 ns/op for libc `malloc` (the malloc
number varies with tcache state and run-to-run), an order of magnitude
gap that holds across runs. The gap exists because the bump path is a
few arithmetic ops and a store, while malloc has to walk buckets and
maintain metadata.

`tests/extreme/test_barrage_extreme.c` measures per-call latency with
`clock_gettime(CLOCK_MONOTONIC)` around each individual `barrage_alloc`
(1M samples in a 16 MiB arena, resetting on exhaustion). Result:
p50=23 ns, p99=27 ns. The clock_gettime call itself costs ~20-30 ns on
x86_64 (vDSO), so p50 is the timer floor, not the true alloc latency;
the amortized bench number above is the honest per-op cost. The p99
tail and the (rare) max reflect timer variance and the occasional
context switch, not allocator work.

## Non-goals

- No growth.
- No per-allocation free.
- No finalizers.
- No internal thread-safety.
- No `madvise` from `barrage_reset` (caller's responsibility).
