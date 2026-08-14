# libsva: API (v0.3)

Status: v0.3 shipped. The v0.2 surface (`SVA_PROT_GUARD_BOTH`, `SVA_PROT_HUGETLB`,
`sva_underflow_guard_addr()`, the pluggable bookkeeping `sva_allocator_t`,
`sva_flush_tlb()`, the W^X fallback, and the `sva_round_up` overflow guard)
is part of the stable API. v0.3 adds no new symbols; the changes are
documentation, the `bench_flush_tlb` harness, and the `test_edge` / extreme
coverage that locks the v0.2 contracts.

## Overview

Guarded `mmap` wrapper: a usable region surrounded by `PROT_NONE` guard
page(s) for overflow (always) and underflow (opt-in via
`SVA_PROT_GUARD_BOTH`). Default layout:

    [usable region][overflow guard page]

With `SVA_PROT_GUARD_BOTH`:

    [underflow guard page][usable region][overflow guard page]

With `SVA_PROT_HUGETLB` each guard slot is a full 2 MB huge page instead
of a 4 KB page, and the usable size is rounded up to 2 MB. The two flags
combine; the worst-case `SVA_PROT_GUARD_BOTH | SVA_PROT_HUGETLB` layout
for a 1-byte request is 6 MB / 3 huge pages.

Executable memory support is included (`PROT_EXEC`; the macOS `MAP_JIT`
requirement is deferred alongside general cross-platform work). libsva
does **not** install its own `SIGSEGV` handler. Catching the guard-page
fault is entirely the caller's responsibility, typically via `libcrash`
as a documented integration pattern (no link-time dependency, per the
toolkit-level architecture decision).

## Types

```c
typedef struct sva_region sva_region_t;   // opaque

typedef enum {
    SVA_PROT_READ       = 1 << 0,
    SVA_PROT_WRITE      = 1 << 1,
    SVA_PROT_EXEC       = 1 << 2,
    /* v0.2: install a PROT_NONE guard page *before* the usable region in
     * addition to the overflow guard after it. Layout when set:
     *   [guard page][usable region][guard page]
     * Without the flag, only the overflow guard (after the region) is
     * installed, matching v0.1 behavior. */
    SVA_PROT_GUARD_BOTH = 1 << 3,
    /* v0.2: back the mapping with MAP_HUGETLB. The usable size is rounded
     * up to 2 MB and each guard slot is a full 2 MB huge page (a 4 KB
     * guard before a 2 MB huge page is useless because the huge page's
     * alignment requirement would not let the two pack contiguously
     * without reserving a full 2 MB slot for the guard anyway). */
    SVA_PROT_HUGETLB    = 1 << 4,
} sva_prot_flags_t;

typedef struct {
    void *(*alloc)(size_t size, void *user_data);   // bookkeeping structures only
    void  (*free)(void *ptr, size_t size, void *user_data);
    void  *user_data;
} sva_allocator_t;   // NULL fields = default to libc malloc/free (bookkeeping only; the mmap'd region itself is never routed through this)

typedef enum {
    SVA_OK                 = 0,
    SVA_ERR_INVALID        = -1,
    SVA_ERR_MMAP_FAILED    = -2,
    SVA_ERR_PROTECT_FAILED = -3,
    SVA_ERR_EXEC_DENIED    = -4,   // W^X hardening blocked PROT_EXEC, see below
} sva_err_t;
```

## API

```c
// Maps `size` bytes (rounded up to the alignment implied by the flags: page
// size by default, 2 MB when SVA_PROT_HUGETLB is set) with the given
// protection flags, surrounded by PROT_NONE guard page(s). The default
// layout places one guard page immediately after the usable region
// (overflow guard). SVA_PROT_GUARD_BOTH adds a second guard page
// immediately before the usable region (underflow guard). With
// SVA_PROT_HUGETLB each guard slot is one huge page (2 MB) instead of one
// 4 KB page.
//
// On systems requiring MAP_JIT for executable+writable mappings (macOS,
// deferred to cross-platform work but flagged here so the flag exists in
// the API now), SVA_PROT_EXEC combined with SVA_PROT_WRITE will need that
// handling later.
//
// `allocator` is for libsva's internal bookkeeping structures (the
// `sva_region_t` handle and any per-region metadata). NULL = libc malloc.
// The mmap'd region itself is NEVER routed through this allocator: the
// data pages and guard pages always come from a single mmap() and go back
// via a single munmap() at sva_unmap() time. The allocator only governs
// the ~48-byte sva_region_t control struct.
//
// W^X fallback (Linux): if SVA_PROT_EXEC is requested and the kernel /
// hardening configuration rejects it (mmap fails with EACCES or EPERM,
// e.g. SELinux `execmem` denial, apparmor, or a hardened kernel),
// sva_map_guarded() does NOT fail outright. It retries the mapping with
// SVA_PROT_EXEC dropped (falling back to just the requested READ/WRITE
// flags), returns the resulting non-executable region, and sets
// *out_err = SVA_ERR_EXEC_DENIED so the caller knows exec was silently
// downgraded. The returned region pointer is still valid and usable.
// Other mmap failures (notably ENOMEM when MAP_HUGETLB is requested but
// the huge page pool is empty) are NOT retried; they return NULL with
// SVA_ERR_MMAP_FAILED directly.
//
// Size validation: size==0 is rejected with SVA_ERR_INVALID. A size
// within (alignment - 1) of SIZE_MAX is also rejected with
// SVA_ERR_INVALID because the internal round-up
//   sva_round_up(v, m) = (v + m - 1) & ~(m - 1)
// would otherwise silently wrap to 0 and slip past the size==0 check,
// yielding a degenerate region consisting of only guard page(s). The
// overflow check is `size > SIZE_MAX - (align - 1)`, applied before the
// round-up. A second overflow check on `usable_size + guard_count *
// guard_size` catches sizes that pass the round-up check but would wrap
// the total mmap length; it also returns SVA_ERR_INVALID.
//
// Return / error contract (callers MUST follow this):
//   non-NULL return + *out_err == SVA_OK                -> full success
//   non-NULL return + *out_err == SVA_ERR_EXEC_DENIED   -> success-with-warning (exec silently downgraded)
//   NULL return    + *out_err < 0                       -> hard failure
//
// The success-with-warning case is the only path on which the return
// value is non-NULL AND *out_err != SVA_OK. Callers checking only
// `if (region == NULL)` will miss the silent downgrade; callers checking
// only `if (*out_err < 0)` will treat a successful-but-downgraded mapping
// as a hard failure. Check both. Callers that require executable memory
// unconditionally must treat SVA_ERR_EXEC_DENIED as fatal for their use
// case and sva_unmap() the region.
//
// out_err may be NULL; in that case the function simply does not write the
// status code, but all validation and error paths still execute (so a NULL
// return still means failure). This is exercised by test_edge.
sva_region_t *sva_map_guarded(size_t size, sva_prot_flags_t flags,
                              const sva_allocator_t *allocator /* nullable */,
                              sva_err_t *out_err /* nullable */);

// Unmaps the region (usable bytes plus all guard pages, via a single
// munmap of the original total mmap length) and frees the sva_region_t
// bookkeeping struct via the allocator recorded at map time. The
// allocator is cached in a local before the free, since the struct is
// what's being freed. Passing NULL is a no-op.
void sva_unmap(sva_region_t *region);

// Returns the usable base pointer (the guard page(s) are not included and
// are not caller-addressable; touching one is the overflow/underflow
// condition this module exists to detect). Returns NULL if region is NULL.
void *sva_base(const sva_region_t *region);

// Returns the usable size (excludes all guard pages). Returns 0 if region
// is NULL. With SVA_PROT_HUGETLB the returned size is the rounded-up 2 MB
// multiple, not the size the caller passed in.
size_t sva_size(const sva_region_t *region);

// Returns the address of the overflow guard page (the PROT_NONE page
// immediately after the usable region). Always non-NULL on a valid
// region. Mainly useful for a caller's own SIGSEGV handler to check "did
// this fault come from one of my guarded regions" by address-range
// comparison. With SVA_PROT_GUARD_BOTH the region also has an underflow
// guard page; use sva_underflow_guard_addr() to retrieve that one.
void *sva_guard_page_addr(const sva_region_t *region);

// v0.2: returns the address of the underflow guard page (the PROT_NONE
// page installed immediately *before* the usable base when the region was
// created with SVA_PROT_GUARD_BOTH). Returns NULL if the region was not
// created with SVA_PROT_GUARD_BOTH, or if region is NULL. The returned
// address is `sva_base(region) - guard_size` (guard_size is page size or
// 2 MB, matching the alignment in effect at map time).
void *sva_underflow_guard_addr(const sva_region_t *region);

// Flushes the TLB for the given region's usable range. On Linux this is
// implemented as mprotect(base, size, PROT_NONE) followed by
// mprotect(base, size, original_prot); both calls trigger
// flush_tlb_range in the kernel as a side effect of changing the VMA
// protections. invlpg is privileged (ring 0) and is NEVER used from
// userspace. Cost is two syscalls plus the kernel-side shootdown.
//
// The original protection is cached in region->prot at map time and is
// restored verbatim. region->prot excludes MAP_HUGETLB (a flag, not a
// prot bit), so the restore call re-applies READ/WRITE/EXEC correctly
// on HUGETLB regions.
//
// The two mprotect calls target only the usable range. The guard page(s)
// are not touched, so they remain PROT_NONE across the flush (this
// matters for SVA_PROT_GUARD_BOTH regions: the flush must not punch a
// hole in either guard). test_edge verifies post-flush that both the
// underflow and overflow guards still SIGSEGV.
//
// Returns 0 on success, SVA_ERR_INVALID if region is NULL, or
// SVA_ERR_PROTECT_FAILED if either mprotect call fails.
//
// Most callers do NOT need this: mprotect, munmap, and mremap already
// flush the TLB as a side effect. Use this only when you've changed PTEs
// without going through those syscalls (rare in userspace).
int sva_flush_tlb(sva_region_t *region);
```

## Non-goals

- No `MAP_HUGETLB` pool management. The entire `[guard(s) + usable +
  guard]` range is mapped as a single `MAP_HUGETLB` mapping, so each 2 MB
  guard slot initially consumes a huge-page reservation from the system
  pool. Worst case (`SVA_PROT_GUARD_BOTH | SVA_PROT_HUGETLB`, 2 MB usable)
  is 6 MB / 3 huge pages, a 200% pool overhead vs. the 2 MB the caller
  asked for. Whether the kernel releases the physical huge page after the
  `mprotect(PROT_NONE)` split is kernel-version/config-dependent.
  Reducing this overhead would require separate mmap calls for usable
  range and guards joined by `MAP_FIXED_NOREPLACE`; deferred.
- No built-in SIGSEGV handling or auto-growth-on-fault (stack-like
  expansion). This is purely a mapping primitive; the "coroutine stack
  with guard page, caught by libcrash" pattern lives in
  documentation/examples, not in this module's code.
- No protection against another component's `mmap(MAP_FIXED)` into the
  guard range. `MAP_FIXED` silently unmaps existing mappings; if another
  library or the same process punches a hole in a guard page this way,
  the guard is bypassed for the unmapped range. libsva does not lock its
  VMAs against this and does not install an in-process watcher. The
  recommended mitigation is the same as for any `MAP_FIXED` hazard: do
  not let unrelated code pick addresses inside regions it does not own.
  This is a deliberate non-goal; the API surface stays small.
- No Windows (`VirtualAlloc`/`VirtualProtect`) support. Linux
  `mmap`/`mprotect` only, matching the toolkit-level platform decision.
  `MAP_JIT` (macOS) handling deferred alongside general cross-platform
  work.
- No `userfaultfd(2)` integration. userfaultfd is a separate mechanism
  for user-space paging and is out of scope.
- No fixed-address API. If a fixed-address overload is added later, it
  MUST use `MAP_FIXED_NOREPLACE` (Linux 4.17+, returns `EEXIST` on
  conflict), never bare `MAP_FIXED` (which silently unmaps existing
  mappings). Bare `MAP_FIXED` is also the bypass vector above and is
  explicitly not mitigated.
