; libcrash x86-64 ASM helpers (NASM, elf64).
;
; Two leaf routines called from the async-signal-safe C signal handler:
;   crash_rdtsc       -- read the timestamp counter, return as uint64_t.
;   crash_copy_4k     -- copy 4096 bytes via SSE2 movdqu, used for the
;                        stack snapshot. Avoids memcpy, which is not on
;                        the POSIX async-signal-safe list (signal-safety(7)).
;
; crash_copy_4k uses SSE2 movdqu (128-bit loads/stores), which is
; baseline x86-64 (every x86-64 CPU since 2003). The previous version
; used AVX vmovdqu (256-bit), which faulted with #UD on CPUs without
; AVX (e.g. Intel Goldmont Plus / Celeron N4100).
;
; Both routines are leaf functions (no calls). crash_copy_4k preserves
; every GPR except rcx (loop counter, caller-saved) and xmm0 (scratch).
; crash_rdtsc returns the TSC in rax and clobbers rdx; both are
; caller-saved per the SysV AMD64 ABI.

default rel

section .text

global crash_rdtsc
global crash_copy_4k

; uint64_t crash_rdtsc(void)
;   Returns the 64-bit timestamp counter in rax.
crash_rdtsc:
    rdtsc                       ; EDX:EAX = TSC
    shl   rdx, 32
    or    rax, rdx
    ret

; void crash_copy_4k(void *dst, const void *src)
;   rdi = dst, rsi = src. Copies exactly 4096 bytes (256 * 16-byte XMM
;   loads/stores). If src is unmapped, the load faults and the kernel
;   re-enters the signal handler; the re-entry guard catches it.
crash_copy_4k:
    xor     rcx, rcx
.loop:
    movdqu  xmm0, [rsi + rcx]
    movdqu  [rdi + rcx], xmm0
    add     rcx, 16
    cmp     rcx, 4096
    jb      .loop
    ret
