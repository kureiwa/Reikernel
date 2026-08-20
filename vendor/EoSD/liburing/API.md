# liburing: API (v0.1, shipped)

Status: shipped. Implementation in `src/uring.c` + `src/uring_x86_64.asm`
(x86_64 only, System V AMD64 ABI). Tests in `tests/`, benches in `bench/`.
All signatures match `include/uring.h`. Built and verified against
kernel 5.10.134, gcc 14.2.0, nasm 2.16.01.

## Overview

A thin, low-level wrapper around the Linux io_uring interface. Provides
SQE (submission queue entry) setup, CQE (completion queue entry)
polling, and the `io_uring_enter(2)` syscall. No event loop, no
scheduler, no integration with other EoSD modules. The caller batches
submissions and invokes `uring_enter` explicitly; the caller polls and
reaps completions via `uring_reap_cqe` / `uring_drain_cqes`.

Scope is deliberately narrow: nop, read, write. SQE setup helpers for
other opcodes (readv, writev, accept, send, recv, etc.) are not
exposed; callers needing them can extend the prep helper set locally
or call `io_uring_enter` with hand-built SQEs. The library's value is
the ring setup, the mmap layout, the memory ordering, and the
single-submitter / single-reaper fast paths -- not a comprehensive
opcode wrapper.

Requires kernel 5.1+ (io_uring was added in 5.1). Uses
`IORING_FEAT_SINGLE_MMAP` (5.3+) when advertised to coalesce the SQ
and CQ ring into a single mmap. Falls back to two separate mmaps on
5.1-5.2.

## Types

```c
typedef struct uring uring_t;   /* opaque */

typedef enum {
    UREING_OK            = 0,
    UREING_ERR_INVALID   = -1,  /* NULL handle, non-power-of-two entries */
    UREING_ERR_SETUP     = -2,  /* io_uring_setup(2) failed (ENOSYS, EINVAL, EMFILE) */
    UREING_ERR_MMAP      = -3,  /* mmap of one of the three ring regions failed */
    UREING_ERR_SQ_FULL   = -4,  /* submission queue full (no free SQE) */
    UREING_ERR_CQ_EMPTY  = -5,  /* completion queue empty at the head index */
} uring_err_t;

#define UREING_ENTER_GETEVENTS  0x1u   /* IORING_ENTER_GETEVENTS */
#define UREING_ENTER_SQ_WAKEUP  0x2u   /* IORING_ENTER_SQ_WAKEUP  */
#define UREING_ENTER_SQ_WAIT    0x4u   /* IORING_ENTER_SQ_WAIT    */
```

Error codes stay in negative integer space so callers can uniformly
check `if (rc < 0)` per EoSD-SPEC.md section 4. The `UREING_` prefix
is intentional: the public symbol namespace is `uring_*`; the error
enum constants were named to avoid colliding with a possible future
`URING_` macro from the kernel header on systems that ship it.

## API

```c
uring_t *uring_create(unsigned entries, uring_err_t *err);
void     uring_destroy(uring_t *r);

int      uring_prep_nop  (uring_t *r, uint64_t user_data);
int      uring_prep_read (uring_t *r, int fd, void *buf, unsigned len,
                          uint64_t offset, uint64_t user_data);
int      uring_prep_write(uring_t *r, int fd, const void *buf, unsigned len,
                          uint64_t offset, uint64_t user_data);

int      uring_enter(uring_t *r, unsigned to_submit, unsigned min_complete,
                     unsigned flags);

int      uring_reap_cqe   (uring_t *r, int *res, uint64_t *user_data);
unsigned uring_drain_cqes (uring_t *r, int *res_out, uint64_t *user_data_out,
                           unsigned max_count);

unsigned uring_sq_pending(const uring_t *r);
unsigned uring_cq_ready  (const uring_t *r);
```

### uring_create / uring_destroy

```c
uring_t *uring_create(unsigned entries, uring_err_t *err);
void     uring_destroy(uring_t *r);
```

`uring_create` calls `io_uring_setup(entries, &params)`, then mmaps
three regions per the io_uring ABI:

- **SQ ring** at `IORING_OFF_SQ_RING`. Contains the SQ head, tail,
  ring_mask, ring_entries, flags, dropped, and the `array` of `__u32`
  indices into the SQE array.
- **CQ ring** at `IORING_OFF_CQ_RING`. Contains the CQ head, tail,
  ring_mask, ring_entries, overflow, and the `cqes` array of
  `struct io_uring_cqe`. When `IORING_FEAT_SINGLE_MMAP` is advertised
  (kernel 5.3+), this region aliases the SQ ring mapping and only one
  `mmap` is issued.
- **SQE array** at `IORING_OFF_SQES`. `sq_entries` `struct io_uring_sqe`
  entries, each 64 bytes in the standard layout.

`entries` must be a power of two and `>= 1`. The kernel rounds up to
the next power of two internally; we reject non-power-of-two values
locally so the caller gets a deterministic `UREING_ERR_INVALID` rather
than a silent resize. The CQ ring is sized by the kernel (typically
`2 * entries`) and reported in `params.cq_entries` after the setup
syscall returns; the library caches it as `r->cq_entries`.

Returns a heap-allocated `uring_t *` on success and, if `err` is
non-NULL, writes `UREING_OK`. Returns NULL and writes one of
`UREING_ERR_INVALID`, `UREING_ERR_SETUP`, `UREING_ERR_MMAP` to `*err`
on failure. `err` may be NULL.

`uring_destroy` munmaps the three regions, closes the io_uring fd,
and frees the handle. NULL is a no-op. Pending SQEs and unreaped CQEs
are lost; the kernel does not flush them on close.

**ENOSYS handling.** If `io_uring_setup` returns `ENOSYS` (kernel <
5.1 or io_uring compiled out), `uring_create` returns NULL with
`*err = UREING_ERR_SETUP`. The shipped tests and benches detect this
case and `printf("SKIP: io_uring not available\n"); return 0;` so
`make test` does not fail on sandboxed or older kernels. Applications
that require io_uring should check the return value and fall back to
`epoll` / blocking I/O when `err == UREING_ERR_SETUP` and
`errno == ENOSYS`.

### uring_prep_nop / uring_prep_read / uring_prep_write

```c
int uring_prep_nop  (uring_t *r, uint64_t user_data);
int uring_prep_read (uring_t *r, int fd, void *buf, unsigned len,
                     uint64_t offset, uint64_t user_data);
int uring_prep_write(uring_t *r, int fd, const void *buf, unsigned len,
                     uint64_t offset, uint64_t user_data);
```

Queues a single SQE. `uring_prep_nop` issues `IORING_OP_NOP`, which
completes inline immediately when `io_uring_enter` is called.
`uring_prep_read` issues `IORING_OP_READ` (the `read(2)`-style
single-buffer variant, not `IORING_OP_READV`). `uring_prep_write`
issues `IORING_OP_WRITE`.

`offset` is a byte offset for regular files; pass 0 for sockets and
pipes (the kernel ignores it for stream-like fds). `user_data` is an
opaque 64-bit cookie echoed back in the matching CQE; the caller uses
it to route completions back to the originating request.

None of these call `io_uring_enter`. The caller batches one or more
SQEs via repeated `uring_prep_*` calls, then invokes `uring_enter`
once with `to_submit = uring_sq_pending(r)` to push them all to the
kernel in a single syscall.

Returns 0 on success, `UREING_ERR_SQ_FULL` if the SQ has no free slot
(the caller must `uring_enter` to drain the SQ before prepping more),
`UREING_ERR_INVALID` if `r` is NULL or `buf` is NULL with `len > 0`.

**SQ publication order.** Each `uring_prep_*` writes the SQE fields
to the next free slot in the SQE array (`sqes[sq_tail & mask]`), then
writes the SQE index into `sq_array[sq_tail & mask]`, then
release-stores `sq_tail + 1`. The release store pairs with the
kernel's acquire load of `sq_tail`. The SQE field writes and the
`sq_array` write are visible to the kernel before it sees the new
tail.

### uring_enter

```c
int uring_enter(uring_t *r, unsigned to_submit, unsigned min_complete,
                unsigned flags);
```

Direct passthrough to `io_uring_enter(ring_fd, to_submit, min_complete,
flags, NULL, 0)`. The `sig` and `sigsz` arguments are always NULL / 0;
callers needing signal-mask handling during `GETEVENTS` waits should
extend the API locally or use `IORING_ENTER_EXT_ARG` directly.

`to_submit` is the number of SQEs the caller believes are pending
(typically `uring_sq_pending(r)`). The kernel consumes up to that
many. `min_complete` is the minimum number of CQEs to wait for, only
meaningful when `flags` includes `UREING_ENTER_GETEVENTS`. `flags` is
a bitwise OR of `UREING_ENTER_*`.

Returns the value returned by `io_uring_enter(2)`: the number of SQEs
actually submitted (>= 0) on success, or a negative errno-style code
on failure. Callers can distinguish "submitted fewer than requested"
(a non-negative return < `to_submit`) from a hard error (negative
return).

### uring_reap_cqe / uring_drain_cqes

```c
int      uring_reap_cqe  (uring_t *r, int *res, uint64_t *user_data);
unsigned uring_drain_cqes(uring_t *r, int *res_out, uint64_t *user_data_out,
                          unsigned max_count);
```

`uring_reap_cqe` reads the CQ entry at the head index (acquire load
on `cq_head` pairing with the kernel's release store on `cq_tail`),
copies out `res` and `user_data`, and advances the head with a release
store pairing with the kernel's acquire load of `cq_head`.

Returns 0 on success, `UREING_ERR_CQ_EMPTY` if no CQE is ready at the
head, `UREING_ERR_INVALID` if `r` is NULL. Output pointers may be
NULL (the value is then not written). On `UREING_ERR_CQ_EMPTY`, the
output pointers are not touched.

`uring_drain_cqes` reaps up to `max_count` CQEs into caller-provided
arrays. `res_out` and `user_data_out` must each point to at least
`max_count` elements. Either may be NULL (the corresponding field is
then not written). Returns the number of CQEs reaped (0..`max_count`).
Stops at the first empty slot. A single release store of the advanced
head is issued at the end of the batch, so the cost is amortized over
the batch.

### uring_sq_pending / uring_cq_ready

```c
unsigned uring_sq_pending(const uring_t *r);
unsigned uring_cq_ready  (const uring_t *r);
```

Snapshots of the queue depths. `uring_sq_pending` returns
`sq_tail - sq_head` (SQEs queued but not yet consumed by the kernel).
`uring_cq_ready` returns `cq_tail - cq_head` (CQEs posted by the
kernel but not yet reaped). Both return 0 if `r` is NULL. The values
may change before the caller inspects them; they are advisory for
batching decisions, not synchronization primitives.

## Thread-safety

The io_uring SQ and CQ rings are NOT multi-producer / multi-consumer
in the liburing configuration. The library assumes a single submitter
thread and a single reaper thread (which may be the same thread).
Concurrent `uring_prep_*` calls on the same handle race on `sq_tail`;
concurrent `uring_reap_cqe` / `uring_drain_cqes` calls race on
`cq_head`. Callers needing multi-threaded submission or reaping must
serialize externally (a spinlock around batches is the typical
pattern; see libspinit).

`uring_create` and `uring_destroy` are safe to call concurrently with
other `uring_create` calls; each returns an independent ring. The
caller must ensure no concurrent submit / reap on the same handle
during `uring_destroy`.

`uring_sq_pending` and `uring_cq_ready` are safe to call from any
thread.

## Error handling

All public functions return 0 on success and a negative `UREING_ERR_*`
code on failure. `uring_enter` is the exception: it returns the raw
`io_uring_enter(2)` return value (a non-negative count or a negative
errno) rather than mapping to `UREING_ERR_*`, because the count is
informational and callers typically want to know how many SQEs were
submitted. A negative return from `uring_enter` is a system errno
(NEGATIVE, in the same `if (rc < 0)` space as the other errors).

`uring_drain_cqes` returns a count (0..`max_count`), never an error
code; an empty CQ reports as 0, not as `UREING_ERR_CQ_EMPTY`.

## Minimal usage example

```c
#include <uring.h>
#include <stdio.h>

int main(void)
{
    uring_err_t err = UREING_OK;
    uring_t *r = uring_create(8, &err);
    if (!r) {
        fprintf(stderr, "uring_create: err=%d\n", err);
        return 1;
    }

    uring_prep_nop(r, 0xCAFEBABE);
    uring_enter(r, 1, 1, UREING_ENTER_GETEVENTS);

    int      res;
    uint64_t ud;
    uring_reap_cqe(r, &res, &ud);
    printf("nop: res=%d user_data=0x%llx\n", res, (unsigned long long)ud);

    uring_destroy(r);
    return 0;
}
```

## Non-goals

- **No event loop.** The caller drives submission and reaping. No
  internal poller thread, no callback dispatch.
- **No scheduler.** No integration with libtick, libspoon, or any
  coroutine / fiber system. Such integration is a documented pattern
  in DESIGN.md, not library code.
- **No SQPOLL.** The `IORING_SETUP_SQPOLL` flag (kernel-side SQ poll
  thread) is not exposed. Callers that want SQPOLL can extend the
  setup flags locally.
- **No buffer / file registration.** `IORING_REGISTER_BUFFERS` and
  `IORING_REGISTER_FILES` are not wrapped. Callers needing fixed
  buffers / files can call `io_uring_register(2)` directly on the
  ring fd (exposed via a future `uring_fd(r)` accessor if needed).
- **No SQE128 / CQE32.** The standard 64-byte SQE / 16-byte CQE
  layout is assumed. `IORING_SETUP_SQE128` and `IORING_SETUP_CQE32`
  are not requested.
- **No Windows / macOS support in v0.1.** io_uring is Linux-only; the
  NASM file is `elf64` only. A future port would require either
  `kqueue`/IOCP backends (substantial rework) or remaining
  Linux-only.
- **No timeout / cancel / accept / send / recv wrappers in v0.1.**
  Only nop, read, write. The SQE setup helper set is easily
  extensible; callers needing more opcodes can copy the pattern from
  `uring_prep_read`.
