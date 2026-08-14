/*
 * liburing implementation. See include/uring.h for the public contract
 * and DESIGN.md for the rationale.
 *
 * io_uring ABI summary (kernel 5.1+; verified against
 * /usr/include/linux/io_uring.h on this host):
 *
 *   io_uring_setup(entries, &params) -> fd
 *       On success, params.sq_entries, params.cq_entries, params.sq_off,
 *       and params.cq_off are filled. params.features (5.3+) advertises
 *       IORING_FEAT_SINGLE_MMAP: the SQ ring and CQ ring can share a
 *       single mmap at IORING_OFF_SQ_RING. We handle both cases (single
 *       and separate) but the kernel has set SINGLE_MMAP since 5.3, so
 *       the separate path is exercised only on 5.1-5.2 or if the feature
 *       bit is somehow absent.
 *
 *   mmap(0, size, PROT_R|W, MAP_SHARED|MAP_POPULATE, fd, IORING_OFF_SQ_RING)
 *       -> SQ ring region. Contains the SQ head, tail, mask, array.
 *       Offsets within this region are params.sq_off.{head, tail, ring_mask,
 *       ring_entries, flags, dropped, array}.
 *
 *   mmap(0, size, PROT_R|W, MAP_SHARED|MAP_POPULATE, fd, IORING_OFF_CQ_RING)
 *       -> CQ ring region. Contains the CQ head, tail, mask, overflow,
 *       cqes. Offsets: params.cq_off.{head, tail, ring_mask, ring_entries,
 *       overflow, cqes}.
 *
 *   mmap(0, size, PROT_R|W, MAP_SHARED|MAP_POPULATE, fd, IORING_OFF_SQES)
 *       -> SQE array. params.sq_off.array holds the offset (within the SQ
 *       ring region) of the array of __u32 indices into this SQE array.
 *       With IORING_SETUP_NO_SQARRAY (5.19+) the index array is gone and
 *       SQ tail indexes SQEs directly; we do not set that flag, so the
 *       indirection is always present.
 *
 *   io_uring_enter(fd, to_submit, min_complete, flags, sig, sigsz)
 *       Submits up to `to_submit` SQEs from the SQ tail, optionally waits
 *       for `min_complete` CQEs if IORING_ENTER_GETEVENTS is set. Returns
 *       the number of SQEs submitted (>= 0) or a negative errno.
 *
 * Memory ordering (ring is shared between userspace and kernel):
 *
 *   SQ: kernel reads tail (acquire), we write tail (release).
 *       We read head (acquire), kernel writes head (release).
 *   CQ: kernel writes tail (release), we read tail (acquire).
 *       We write head (release), kernel reads head (acquire).
 *
 * On x86_64 these compile to plain MOVs (TSO). The C11 atomics are
 * required for correctness on weakly-ordered architectures (future
 * ARM64 port) and for documentation. The dedicated uring_barrier()
 * symbol in uring_x86_64.asm is provided for callers that want an
 * explicit full memory barrier (sfence + lfence) without going through
 * stdatomic; the library itself uses stdatomic for all ring-pointer
 * accesses.
 */

#define _GNU_SOURCE
#include <uring.h>

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <errno.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#include <linux/io_uring.h>

/* x86_64-only asm helper. The C11 atomics below are the source of truth
 * for all ring-pointer accesses. uring_barrier (sfence + lfence) is
 * provided as a stable C-callable symbol for benchmarks and for callers
 * that want an explicit io_uring kernel-shared-memory barrier without
 * going through stdatomic. The asm symbol is in src/uring_x86_64.asm;
 * porting to another OS/arch requires porting the asm too. */
extern void uring_barrier(void);

/* io_uring_setup(2) is not wrapped by glibc; we invoke it via syscall(2).
 * io_uring_enter(2) IS wrapped by modern glibc (>= 2.35) but the wrapper
 * is a thin syscall passthrough, so calling syscall(SYS_io_uring_enter,
 * ...) directly works on all glibc versions. We use the raw syscall for
 * both for portability with older glibc. */
static int uring_setup_syscall(unsigned entries, struct io_uring_params *p)
{
    return (int)syscall(SYS_io_uring_setup, (long)entries, p);
}

static int uring_enter_syscall(int fd, unsigned to_submit, unsigned min_complete,
                               unsigned flags)
{
    return (int)syscall(SYS_io_uring_enter, (long)fd, (long)to_submit,
                        (long)min_complete, (long)flags, NULL, 0);
}

/*
 * Ring handle. The three mmap'd regions are tracked separately so we can
 * munmap them in uring_destroy without computing sizes from offsets.
 *
 *   sq_ring / cq_ring: the SQ and CQ ring control regions. With
 *     IORING_FEAT_SINGLE_MMAP they alias the same mapping; in that case
 *     sq_ring_base == cq_ring_base and we munmap only once.
 *   sqes: the SQE array. Always a separate mmap at IORING_OFF_SQES.
 *
 *   The "*_off" fields cache the offsets reported by io_uring_setup so
 *   the prep/reap fast paths do not have to chase through params on every
 *   call. The kernel reports these offsets as byte offsets within the
 *   ring region; we add them to the ring base pointer at access time.
 *
 *   sqe_size is sizeof(io_uring_sqe) for the standard 64-byte SQE layout.
 *     IORING_SETUP_SQE128 (5.19) doubles it; we do not request that flag.
 */
struct uring {
    int      ring_fd;
    void    *sq_ring_base;   /* mmap base at IORING_OFF_SQ_RING */
    size_t   sq_ring_size;
    void    *cq_ring_base;   /* mmap base at IORING_OFF_CQ_RING (== sq_ring_base with SINGLE_MMAP) */
    size_t   cq_ring_size;
    void    *sqes_base;      /* mmap base at IORING_OFF_SQES */
    size_t   sqes_size;

    /* Cached offsets (bytes from the respective ring base). */
    uint32_t sq_head_off;
    uint32_t sq_tail_off;
    uint32_t sq_mask_off;
    uint32_t sq_array_off;
    uint32_t cq_head_off;
    uint32_t cq_tail_off;
    uint32_t cq_mask_off;
    uint32_t cq_cqes_off;

    unsigned sq_mask;        /* sq_entries - 1, cached for the SQE-index fast path */
    unsigned cq_mask;        /* cq_entries - 1 */
    unsigned sq_entries;
    unsigned cq_entries;

    /* 1 if the SQ and CQ ring share one mmap (IORING_FEAT_SINGLE_MMAP).
     * Controls whether uring_destroy munmaps sq_ring_base and
     * cq_ring_base separately or as one region. */
    int      single_mmap;
};

static int is_power_of_two(unsigned n)
{
    return n != 0 && (n & (n - 1)) == 0;
}

/*
 * Ring region size computation. The SQ ring region needs to span from
 * offset 0 to the end of the SQ index array; the kernel's mmap helper
 * sizes the region to cover max(sq_off.array + sq_entries * 4,
 * cq_off.cqes + cq_entries * sizeof(io_uring_cqe)) when SINGLE_MMAP is
 * in effect (SQ and CQ share one region). Without SINGLE_MMAP the SQ
 * region is sq_off.array + sq_entries * 4 and the CQ region is
 * cq_off.cqes + cq_entries * sizeof(io_uring_cqe).
 *
 * We size each region to cover the highest offset the kernel reports
 * for that region, rounded up to a page. Rounding up is defensive: the
 * kernel accepts a larger size and only the requested pages are mapped.
 */
static size_t ring_region_size(uint32_t last_offset, uint32_t last_stride,
                               size_t bytes_per_entry)
{
    /* `last_offset` is the offset of the last array within the region;
     * `last_stride` is the count of entries in that array; `bytes_per_entry`
     * is the size of one entry. */
    size_t needed = (size_t)last_offset + (size_t)last_stride * bytes_per_entry;
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) {
        ps = 4096;
    }
    size_t page = (size_t)ps;
    /* Round up to a whole number of pages. */
    return (needed + page - 1) & ~(page - 1);
}

static void *map_region(int fd, uint64_t mmap_off, size_t size)
{
    if (size == 0) {
        return NULL;
    }
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_POPULATE, fd, (off_t)mmap_off);
    if (p == MAP_FAILED) {
        return NULL;
    }
    return p;
}

uring_t *uring_create(unsigned entries, uring_err_t *err)
{
    if (err) {
        *err = UREING_OK;
    }
    if (!is_power_of_two(entries)) {
        if (err) *err = UREING_ERR_INVALID;
        return NULL;
    }

    struct io_uring_params p;
    memset(&p, 0, sizeof(p));

    int fd = uring_setup_syscall(entries, &p);
    if (fd < 0) {
        /* ENOSYS: kernel < 5.1 or io_uring compiled out. Tests should
         * skip, not fail. EINVAL: bad entries / unsupported flag.
         * EMFILE/ENFILE: fd exhaustion. All collapse to SETUP. */
        if (err) *err = UREING_ERR_SETUP;
        return NULL;
    }

    uring_t *r = malloc(sizeof(*r));
    if (!r) {
        close(fd);
        if (err) *err = UREING_ERR_SETUP;  /* OOM is a setup-class failure */
        return NULL;
    }
    memset(r, 0, sizeof(*r));
    r->ring_fd = fd;
    r->sq_entries = p.sq_entries;
    r->cq_entries = p.cq_entries;
    r->sq_mask = p.sq_entries - 1;
    r->cq_mask = p.cq_entries - 1;

    /* Cache offsets. These are byte offsets within the respective ring
     * region. With SINGLE_MMAP both rings live in one region and the
     * offsets remain distinct (sq_off.* index into the SQ portion,
     * cq_off.* index into the CQ portion of the same mapping). */
    r->sq_head_off  = p.sq_off.head;
    r->sq_tail_off  = p.sq_off.tail;
    r->sq_mask_off  = p.sq_off.ring_mask;
    r->sq_array_off = p.sq_off.array;
    r->cq_head_off  = p.cq_off.head;
    r->cq_tail_off  = p.cq_off.tail;
    r->cq_mask_off  = p.cq_off.ring_mask;
    r->cq_cqes_off  = p.cq_off.cqes;

    int single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
    r->single_mmap = single;

    /* Compute region sizes. For SINGLE_MMAP we mmap one region at
     * IORING_OFF_SQ_RING large enough to cover both SQ and CQ arrays;
     * the CQ ring aliases the same pointer. Without SINGLE_MMAP we
     * mmap two separate regions. */
    size_t sq_size = ring_region_size(p.sq_off.array, p.sq_entries, 4);
    size_t cq_size = ring_region_size(p.cq_off.cqes, p.cq_entries,
                                      sizeof(struct io_uring_cqe));
    if (single) {
        size_t combined = sq_size > cq_size ? sq_size : cq_size;
        void *base = map_region(fd, IORING_OFF_SQ_RING, combined);
        if (!base) {
            uring_destroy(r);
            if (err) *err = UREING_ERR_MMAP;
            return NULL;
        }
        r->sq_ring_base = base;
        r->sq_ring_size = combined;
        r->cq_ring_base = base;
        r->cq_ring_size = combined;
    } else {
        void *sqb = map_region(fd, IORING_OFF_SQ_RING, sq_size);
        if (!sqb) {
            uring_destroy(r);
            if (err) *err = UREING_ERR_MMAP;
            return NULL;
        }
        r->sq_ring_base = sqb;
        r->sq_ring_size = sq_size;
        void *cqb = map_region(fd, IORING_OFF_CQ_RING, cq_size);
        if (!cqb) {
            uring_destroy(r);
            if (err) *err = UREING_ERR_MMAP;
            return NULL;
        }
        r->cq_ring_base = cqb;
        r->cq_ring_size = cq_size;
    }

    /* SQE array is always a separate mmap at IORING_OFF_SQES. */
    size_t sqes_bytes = (size_t)p.sq_entries * sizeof(struct io_uring_sqe);
    void *sqes = map_region(fd, IORING_OFF_SQES, sqes_bytes);
    if (!sqes) {
        uring_destroy(r);
        if (err) *err = UREING_ERR_MMAP;
        return NULL;
    }
    r->sqes_base = sqes;
    r->sqes_size = sqes_bytes;

    return r;
}

void uring_destroy(uring_t *r)
{
    if (!r) {
        return;
    }
    if (r->sqes_base) {
        munmap(r->sqes_base, r->sqes_size);
    }
    if (r->single_mmap) {
        /* sq_ring_base and cq_ring_base alias; munmap once. */
        if (r->sq_ring_base) {
            munmap(r->sq_ring_base, r->sq_ring_size);
        }
    } else {
        if (r->sq_ring_base) {
            munmap(r->sq_ring_base, r->sq_ring_size);
        }
        if (r->cq_ring_base) {
            munmap(r->cq_ring_base, r->cq_ring_size);
        }
    }
    if (r->ring_fd >= 0) {
        close(r->ring_fd);
    }
    free(r);
}

/*
 * Returns a pointer to the next free SQE, or NULL if the SQ is full.
 * Computes (sq_tail - sq_head) against sq_entries; if equal, the SQ is
 * full. The SQ index array stores __u32 indices into the SQE array; we
 * write the SQE at index (tail & mask) into the SQE array, then publish
 * the SQE by writing the index into sq_array[tail & mask] and advancing
 * sq_tail with a release store.
 *
 * The returned pointer is only valid until the next uring_prep_* call
 * (which may advance to the same SQE slot after a wrap). Callers should
 * treat it as a transient pointer and finish writing before the next
 * call.
 */
static struct io_uring_sqe *next_sqe(uring_t *r)
{
    /* Cast through volatile because the kernel updates sq_head concurrently
     * (it does not, in the single-submitter model, but the type system
     * needs to know this is shared memory). We use stdatomic loads for
     * the actual memory ordering. */
    uint32_t head = atomic_load_explicit(
        (_Atomic uint32_t *)((char *)r->sq_ring_base + r->sq_head_off),
        memory_order_acquire);
    uint32_t tail = atomic_load_explicit(
        (_Atomic uint32_t *)((char *)r->sq_ring_base + r->sq_tail_off),
        memory_order_relaxed);
    if (tail - head >= r->sq_entries) {
        return NULL;  /* SQ full */
    }
    uint32_t idx = tail & r->sq_mask;
    struct io_uring_sqe *sqe =
        (struct io_uring_sqe *)((char *)r->sqes_base + (size_t)idx * sizeof(struct io_uring_sqe));
    return sqe;
}

/*
 * Publishes a filled SQE: writes the SQE index into sq_array[tail & mask]
 * and advances sq_tail with a release store. The release store pairs with
 * the kernel's acquire load of sq_tail. The SQE field writes and
 * the sq_array write are visible to the kernel before it sees the new tail.
 */
static void publish_sqe(uring_t *r)
{
    uint32_t tail = atomic_load_explicit(
        (_Atomic uint32_t *)((char *)r->sq_ring_base + r->sq_tail_off),
        memory_order_relaxed);
    uint32_t idx = tail & r->sq_mask;
    /* sq_array is an array of __u32 indices into the SQE array. */
    volatile uint32_t *sq_array =
        (volatile uint32_t *)((char *)r->sq_ring_base + r->sq_array_off);
    sq_array[idx] = idx;
    /* Release store: orders the SQE writes + sq_array write above before
     * the tail advance. */
    atomic_store_explicit(
        (_Atomic uint32_t *)((char *)r->sq_ring_base + r->sq_tail_off),
        tail + 1, memory_order_release);
}

int uring_prep_nop(uring_t *r, uint64_t user_data)
{
    if (!r) {
        return UREING_ERR_INVALID;
    }
    struct io_uring_sqe *sqe = next_sqe(r);
    if (!sqe) {
        return UREING_ERR_SQ_FULL;
    }
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode    = IORING_OP_NOP;
    sqe->fd        = -1;
    sqe->user_data = user_data;
    publish_sqe(r);
    return UREING_OK;
}

int uring_prep_read(uring_t *r, int fd, void *buf, unsigned len,
                    uint64_t offset, uint64_t user_data)
{
    if (!r || (!buf && len > 0)) {
        return UREING_ERR_INVALID;
    }
    struct io_uring_sqe *sqe = next_sqe(r);
    if (!sqe) {
        return UREING_ERR_SQ_FULL;
    }
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode    = IORING_OP_READ;
    sqe->fd        = fd;
    sqe->addr      = (uint64_t)(uintptr_t)buf;
    sqe->len       = len;
    sqe->off       = offset;
    sqe->user_data = user_data;
    publish_sqe(r);
    return UREING_OK;
}

int uring_prep_write(uring_t *r, int fd, const void *buf, unsigned len,
                     uint64_t offset, uint64_t user_data)
{
    if (!r || (!buf && len > 0)) {
        return UREING_ERR_INVALID;
    }
    struct io_uring_sqe *sqe = next_sqe(r);
    if (!sqe) {
        return UREING_ERR_SQ_FULL;
    }
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode    = IORING_OP_WRITE;
    sqe->fd        = fd;
    sqe->addr      = (uint64_t)(uintptr_t)buf;
    sqe->len       = len;
    sqe->off       = offset;
    sqe->user_data = user_data;
    publish_sqe(r);
    return UREING_OK;
}

int uring_enter(uring_t *r, unsigned to_submit, unsigned min_complete,
                unsigned flags)
{
    if (!r) {
        return UREING_ERR_INVALID;
    }
    return uring_enter_syscall(r->ring_fd, to_submit, min_complete, flags);
}

int uring_reap_cqe(uring_t *r, int *res, uint64_t *user_data)
{
    if (!r) {
        return UREING_ERR_INVALID;
    }
    uint32_t head = atomic_load_explicit(
        (_Atomic uint32_t *)((char *)r->cq_ring_base + r->cq_head_off),
        memory_order_relaxed);
    uint32_t tail = atomic_load_explicit(
        (_Atomic uint32_t *)((char *)r->cq_ring_base + r->cq_tail_off),
        memory_order_acquire);
    if (head == tail) {
        return UREING_ERR_CQ_EMPTY;
    }
    uint32_t idx = head & r->cq_mask;
    /* cqes is an array of struct io_uring_cqe; the offset cq_cqes_off
     * is the byte offset of cqes[0] within the CQ ring region. */
    volatile struct io_uring_cqe *cqe =
        (volatile struct io_uring_cqe *)((char *)r->cq_ring_base
                                         + r->cq_cqes_off
                                         + (size_t)idx * sizeof(struct io_uring_cqe));
    int      cqe_res = cqe->res;
    uint64_t cqe_ud  = cqe->user_data;

    if (res)        *res = cqe_res;
    if (user_data)  *user_data = cqe_ud;

    /* Advance head with a release store. Pairs with the kernel's acquire
     * load of cq_head. Our reads of cqe->res / cqe->user_data
     * are complete before the kernel sees the head advance and is free to
     * recycle the slot. */
    atomic_store_explicit(
        (_Atomic uint32_t *)((char *)r->cq_ring_base + r->cq_head_off),
        head + 1, memory_order_release);
    return UREING_OK;
}

unsigned uring_drain_cqes(uring_t *r, int *res_out, uint64_t *user_data_out,
                          unsigned max_count)
{
    if (!r || max_count == 0) {
        return 0;
    }
    uint32_t head = atomic_load_explicit(
        (_Atomic uint32_t *)((char *)r->cq_ring_base + r->cq_head_off),
        memory_order_relaxed);
    uint32_t tail = atomic_load_explicit(
        (_Atomic uint32_t *)((char *)r->cq_ring_base + r->cq_tail_off),
        memory_order_acquire);
    unsigned count = 0;
    while (count < max_count && head != tail) {
        uint32_t idx = head & r->cq_mask;
        volatile struct io_uring_cqe *cqe =
            (volatile struct io_uring_cqe *)((char *)r->cq_ring_base
                                             + r->cq_cqes_off
                                             + (size_t)idx * sizeof(struct io_uring_cqe));
        if (res_out)        res_out[count]        = cqe->res;
        if (user_data_out)  user_data_out[count]  = cqe->user_data;
        head++;
        count++;
    }
    if (count > 0) {
        atomic_store_explicit(
            (_Atomic uint32_t *)((char *)r->cq_ring_base + r->cq_head_off),
            head, memory_order_release);
    }
    return count;
}

unsigned uring_sq_pending(const uring_t *r)
{
    if (!r) {
        return 0;
    }
    uint32_t head = atomic_load_explicit(
        (_Atomic uint32_t *)((char *)r->sq_ring_base + r->sq_head_off),
        memory_order_acquire);
    uint32_t tail = atomic_load_explicit(
        (_Atomic uint32_t *)((char *)r->sq_ring_base + r->sq_tail_off),
        memory_order_acquire);
    return tail - head;
}

unsigned uring_cq_ready(const uring_t *r)
{
    if (!r) {
        return 0;
    }
    uint32_t head = atomic_load_explicit(
        (_Atomic uint32_t *)((char *)r->cq_ring_base + r->cq_head_off),
        memory_order_acquire);
    uint32_t tail = atomic_load_explicit(
        (_Atomic uint32_t *)((char *)r->cq_ring_base + r->cq_tail_off),
        memory_order_acquire);
    return tail - head;
}
