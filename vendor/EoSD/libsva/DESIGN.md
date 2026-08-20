# libsva: Design Notes (v0.3)

## Problem

Raw `mmap`/`mprotect` are low-level and easy to misuse for the common
"give me a region that crashes cleanly on overflow instead of corrupting
adjacent memory" pattern (coroutine stacks, JIT buffers, memory-mapped
files). libsva wraps this into a small, safe-by-construction API.

## Guard page placement: overflow by default, underflow opt-in (v0.2)

v0.1 shipped with one guard page after the usable region (overflow only).
The original rationale was that the primary use case, coroutine stacks for
`libspoon`, is an overflow concern (a stack growing too deep), not
underflow (nothing legitimately writes before the base of a freshly
allocated region); doubling the guard-page overhead for a scenario that
did not need it was not justified.

v0.2 added `SVA_PROT_GUARD_BOTH` as an opt-in flag rather than making it
the default, so existing v0.1 callers keep their layout and overhead.
With the flag the layout is symmetric:

    [PROT_NONE guard][usable region][PROT_NONE guard]

The underflow case turned out to matter for two scenarios surfaced after
v0.1: hand-rolled interpreters that index backwards from a stack pointer
before adjusting it, and JIT trampolines that compute a target address
with an off-by-one subtraction. Both can land one byte before the base;
without an underflow guard such a write silently corrupts whatever the
kernel placed immediately below the mapping. The opt-in flag keeps the
default zero-overhead and makes the symmetric case explicit at the call
site. The v0.3 surface ships this unchanged.

## Why no built-in SIGSEGV handling

This directly mirrors the toolkit-level "zero cross-deps" decision:
`libcrash` already solves async-signal-safe fault handling generally.
Building a second, narrower SIGSEGV handler inside libsva just for guard-
page faults would duplicate that logic and create exactly the kind of
hidden coupling the toolkit-level architecture explicitly avoided.
`sva_guard_page_addr()` and `sva_underflow_guard_addr()` exist
specifically so a caller's own handler (via libcrash or otherwise) can
identify "this fault was a guard-page hit from one of my sva regions" by
address comparison, without libsva needing to know anything about signal
handling itself.

## Address space

x86_64 Linux userspace has access to 48-bit (4-level paging) or 57-bit
(5-level paging, kernel 4.14+ on supported hardware) virtual addresses;
the user half is therefore 47 or 56 bits, with the canonical-address
split at `0x0000_7fff_ffff_ffff` / `0xffff_8000_0000_0000` (48-bit) or
the corresponding 5-level boundaries. `MAP_32BIT` constrains a mapping
to the first 2 GB, useful for trampoline placement (see libdetour) or
for reaching code that uses 32-bit relative addressing.

## TLB flushing from userspace

`invlpg` is privileged (ring 0) and cannot be issued from userspace.
libsva exposes `sva_flush_tlb(sva_region_t *)` which wraps the cheapest
userspace equivalent: `mprotect(base, size, PROT_NONE)` followed by
`mprotect(base, size, original_prot)`. Both calls trigger
`flush_tlb_range` in the kernel as a side effect of changing the VMA
protections. Cost is two syscalls plus the kernel-side shootdown. PCID
(Process Context ID, modern Intel) reduces TLB flush cost on context
switch but does not help with intra-process flush. Most callers do not
need `sva_flush_tlb` at all -- `mprotect`, `munmap`, and `mremap` already
flush the TLB as a side effect.

The original protection is cached in `region->prot` at map time and is
restored verbatim by the second `mprotect`. The two `mprotect` calls
target only the usable range. The guard pages are not touched, so they
remain `PROT_NONE` across the flush (this matters for `SVA_PROT_GUARD_BOTH`
regions: the flush must not punch a hole in either guard). For
`SVA_PROT_HUGETLB` regions, `mprotect` on a sub-range of a `MAP_HUGETLB`
VMA is kernel-supported and splits the VMA at huge-page boundaries;
`region->prot` excludes `MAP_HUGETLB` (which is a `mmap` flag, not a
`prot` bit), so the restore call re-applies the original READ/WRITE/EXEC
protection correctly. `test_edge` verifies post-flush that both the
underflow and overflow guards still SIGSEGV on a `GUARD_BOTH` region.

## Executable memory / JIT support

`SVA_PROT_EXEC` support is included for the JIT buffer use case. The
macOS `MAP_JIT` requirement (deferred alongside general cross-platform
work per the toolkit-level platform decision) is flagged in the API doc
so the flag naming doesn't need to change later. Only the Linux
implementation is written; it will need a macOS-specific branch when that
work happens. See the "W^X handling" section below for how `PROT_EXEC`
mappings rejected by W^X hardening are handled.

## Huge-page backing (v0.2)

`SVA_PROT_HUGETLB` requests `MAP_HUGETLB` for the entire mapping. The
usable size is rounded up to 2 MB (the x86_64 default huge page size;
reading from `/proc/meminfo` at runtime would be more flexible but the
v0.2 spec fixes this at 2 MB), and each guard slot is a full 2 MB huge
page. A 4 KB guard before a 2 MB huge page would be useless: the huge
page's alignment requirement prevents the two from packing contiguously
without reserving a full 2 MB slot for the guard anyway, so making the
guard itself 2 MB costs no extra address space and matches the
alignment. The whole `[guard(s) + usable + guard]` range is a single
`mmap`; `mprotect(PROT_NONE)` on the guard sub-ranges splits the VMA,
which the kernel supports at huge-page boundaries.

The guard slots initially consume huge-page reservations from the system
pool because the entire mapping (usable + guards) is one
`MAP_HUGETLB` `mmap`. `mprotect(PROT_NONE)` on the guard sub-ranges
splits the VMA; whether the physical huge page backing is released is
kernel-version and configuration dependent. Worst case for
`SVA_PROT_GUARD_BOTH | SVA_PROT_HUGETLB` with 2 MB usable is 6 MB / 3
huge pages, a 200% pool overhead vs. the 2 MB the caller asked for.
Reducing this requires separate `mmap` calls for the usable range and
the guards joined by `MAP_FIXED_NOREPLACE`; deferred (see API.md
Non-goals). The huge-page pool size is read from
`/proc/sys/vm/nr_hugepages`; `test_hugetlb`, `bench_hugetlb`, and the
`test_guard_both_hugetlb` sub-case in `test_edge` all SKIP cleanly when
the pool is empty so the suite is green on a default host.

## Size validation and rounding overflow

`sva_map_guarded` rejects `size == 0` with `SVA_ERR_INVALID`. The
internal round-up `sva_round_up(v, m) = (v + m - 1) & ~(m - 1)` silently
wraps to 0 when `v` is within `m - 1` of `SIZE_MAX`, so the function
also rejects `size > SIZE_MAX - (align - 1)` before rounding. Without
that check a caller passing `SIZE_MAX` (or near it) would slip past the
`size == 0` rejection, get `usable_size == 0`, and end up with a
degenerate region consisting of only guard page(s). A second overflow
check on `usable_size + guard_count * guard_size` catches sizes that
pass the round-up check but would wrap the total mmap length; both
checks return `SVA_ERR_INVALID`. `test_edge` exercises both paths with
`SIZE_MAX`, `SIZE_MAX - 1`, `SIZE_MAX - (page - 1)`, and the
`SVA_PROT_HUGETLB` 2 MB-aligned equivalents.

## W^X handling (resolved)

On Linux, if `SVA_PROT_EXEC` is requested and the mapping is rejected by
W^X hardening (observed as `EACCES` or `EPERM` from `mmap`, e.g. SELinux
`execmem` denial, apparmor, or a hardened kernel), `sva_map_guarded`
retries without `PROT_EXEC` and returns the resulting region with
`*out_err = SVA_ERR_EXEC_DENIED` rather than failing outright. The
return value is non-`NULL` and the region is usable as `READ`/`WRITE`
memory; the only signal that exec was downgraded is `*out_err`. A
partially-degraded-but-usable mapping is more useful to most callers
(e.g. a JIT that can fall back to an interpreter, or fail its own
operation with a clear reason) than an unconditional `NULL`. Callers
that require executable memory unconditionally must check `out_err`
themselves and treat `SVA_ERR_EXEC_DENIED` as fatal for their use case.
Other mmap failures (notably `ENOMEM` when `MAP_HUGETLB` is requested
but the huge page pool is empty) are not retried; they return `NULL`
with `SVA_ERR_MMAP_FAILED` directly.

The success-with-warning contract is: **non-`NULL` return AND
`*out_err != SVA_OK`**. `SVA_ERR_EXEC_DENIED` is currently the only such
code. Callers must check both the return value and `*out_err`, not
either in isolation.

## Benchmarks (v0.3 reference host)

Measured on the v0.3 reference host (gcc 14.2.0, make 4.4.1,
`-std=c11 -Wall -Wextra -Werror -pedantic -O2`). Numbers are typical,
not contractual -- `mmap`/`mprotect` cost is dominated by kernel work and
varies with kernel build, frequency scaling, and TLB state.

- `bench_map` (1000 iterations, 4 KB `MAP_PRIVATE|MAP_ANONYMOUS` map +
  unmap, default guard): **~1123 ns/op**.
  - Two syscalls (`mmap` + `munmap`) + one `mprotect` (overflow guard)
    + one `malloc`/`free` for the `sva_region_t` control struct.
- `bench_flush_tlb` (10000 iterations, 4 KB region, warm): **~561 ns/op
  for `sva_flush_tlb`**, ~542 ns/op for a bare `mprotect(PROT_NONE)` +
  `mprotect(PROT_READ|PROT_WRITE)` pair on the same size. Wrapper
  overhead (NULL check + two field reads + return) is ~6 ns/op.
  - Each call is two `mprotect` syscalls plus the kernel-side shootdown;
    the wrapper logic is negligible vs. the syscall pair.
- `bench_hugetlb` (regular 4 KB path, 200 iterations): ~1135 ns/op,
  consistent with `bench_map`. The 2 MB `MAP_HUGETLB` path is SKIPped
  when `/proc/sys/vm/nr_hugepages == 0`.
- `tests/extreme/test_sva_extreme` (`sva_map_guarded` + `sva_unmap` 4 KB,
  1000 samples, `clock_gettime(CLOCK_MONOTONIC)`):
  **p50 = ~1010 ns**, p99 = ~4180 ns, max = ~10500 ns. The p50/p99 gap
  reflects kernel allocator jitter and is not a libsva-side regression.

`bench_flush_tlb` was added in v0.3 alongside `bench_dump` (libcrash)
and `bench_dummy` (libpmu) to bring libsva's bench surface up to parity
with the rest of the toolkit.

## Non-goals

- No `MAP_HUGETLB` pool management (see "Huge-page backing" above).
- No built-in fault handling.
- No auto-growing regions (stack-like expansion on guard-page fault). A
  region is a fixed size for its lifetime; growth would require unmapping
  and remapping, which callers can do themselves by creating a new,
  larger region if needed.
- No protection against another component's `mmap(MAP_FIXED)` into a
  guard range. `MAP_FIXED` silently unmaps existing mappings; an outside
  `MAP_FIXED` into a guard page bypasses the guard for the unmapped
  range. libsva does not lock its VMAs against this (see API.md
  Non-goals).
- No `userfaultfd(2)` integration (out of scope).
- No fixed-address API; if added later, MUST use `MAP_FIXED_NOREPLACE`
  (see API.md).
- No Windows/macOS support.

## macOS MAP_JIT (deferred, unchanged)

Still deferred to cross-platform expansion alongside general Windows/macOS
work. The Linux implementation uses plain `mmap` with `PROT_EXEC` and no
`MAP_JIT`-equivalent flag, since that's a macOS-specific requirement not
applicable to the current platform target.
