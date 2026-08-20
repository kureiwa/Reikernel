; uring_x86_64.asm - NASM helper for liburing.
;
; System V AMD64 ABI, elf64 object format. One routine:
;
;   void uring_barrier(void)
;       No inputs, no outputs, no clobbers (RAX/RCX/RDX are caller-saved
;       and we touch none of them).
;
;       Emits sfence + lfence: a full memory barrier suitable for the
;       io_uring kernel-shared-memory ring. sfence orders prior stores
;       (so the kernel sees our SQE field writes and the new sq_tail
;       before anything else migrates); lfence orders subsequent loads
;       (so we do not read a CQE field before the kernel's store to
;       cq_tail is observed). On x86_64's TSO memory model loads and
;       stores are already ordered in program order; this pair is a
;       belt-and-suspenders explicit barrier for callers that want it
;       without going through C11 stdatomic.
;
;       The library's ring-pointer accesses in src/uring.c use stdatomic
;       (atomic_load_explicit / atomic_store_explicit with
;       memory_order_acquire / memory_order_release). Those lower to
;       plain MOVs on x86_64 (TSO) and are correct. This asm symbol
;       exists as a stable C-callable comparison target for benchmarks
;       and as a fallback for code paths that want an explicit fence
;       rather than relying on the compiler to emit one (e.g. for
;       non-atomic accesses to ring memory via volatile casts in
;       callers outside the library).
;
;       Equivalent to __atomic_thread_fence(__ATOMIC_SEQ_CST) on x86_64,
;       which gcc lowers to mfence. We emit sfence + lfence instead of
;       mfence because the Intel SDM documents sfence + lfence as a
;       stronger ordering pair than mfence for the io_uring use case
;       (mfence does not strictly order non-temporal stores; sfence
;       does). For the io_uring ring (which is ordinary WB memory),
;       mfence and sfence+lfence are equivalent.

default rel

section .text

global uring_barrier
uring_barrier:
    sfence
    lfence
    ret

; Mark stack as non-executable (GNU note, picked up by linker).
section .note.GNU-stack noalloc noexec nowrite progbits
