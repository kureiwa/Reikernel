; flume_x86_64.asm - NASM helpers for libflume.
;
; System V AMD64 ABI, elf64 object format. Two routines:
;   uint64_t flume_xadd_uint64(_Atomic uint64_t *p, uint64_t inc)
;                                  - LOCK XADD [rdi], rsi; returns old *p
;                                    in rax. This is the wait-free producer
;                                    primitive: a single locked RMW orders
;                                    this producer against every other.
;   void flume_copy_56(void *dst, const void *src)
;                                  - 56-byte unaligned copy via movdqu +
;                                    movq (SSE2, baseline x86_64, no AVX
;                                    needed). Used for the full-size
;                                    enqueue/dequeue/drain fast path.
;
; C11 atomic_fetch_add would also lower to LOCK XADD; the asm wrapper
; exists to make the producer fast path visible in the disassembly and
; to give the bench a stable comparison target (asm-XADD vs C11-XADD).
;
; 56 = 16 + 16 + 16 + 8. The slot's data[] field starts at offset 8
; within a 64-byte aligned slot, so neither src nor dst is 16-byte
; aligned in the enqueue path. movdqu handles unaligned operands; movq
; for the trailing 8 bytes is also unaligned-safe on x86_64.

default rel

section .text

global flume_xadd_uint64
flume_xadd_uint64:
    ; in:  rdi = p (pointer to uint64_t), rsi = inc
    ; out: rax = old *p
    ; LOCK XADD adds inc to *p and returns the OLD value of *p in the
    ; destination register (rax here). rax and rsi are both caller-saved
    ; so no register preservation is needed.
    mov     rax, rsi
    lock xadd [rdi], rax
    ret

global flume_copy_56
flume_copy_56:
    ; in:  rdi = dst, rsi = src
    ; out: 56 bytes copied from src to dst
    ; xmm0..xmm2 are caller-saved (System V AMD64); no need to preserve.
    ; rax is caller-saved; used as a scratch for the trailing 8 bytes.
    ; movdqu for the three 16-byte chunks (SSE2, unaligned). Plain
    ; `mov` for the trailing 8 bytes: NASM auto-sizes from the 64-bit
    ; GPR, avoiding the `movq` ambiguity with the MMX/SSE opcode of
    ; the same mnemonic.
    movdqu  xmm0, [rsi +  0]
    movdqu  xmm1, [rsi + 16]
    movdqu  xmm2, [rsi + 32]
    mov     rax,  [rsi + 48]
    movdqu  [rdi +  0], xmm0
    movdqu  [rdi + 16], xmm1
    movdqu  [rdi + 32], xmm2
    mov     [rdi + 48], rax
    ret

; Mark stack as non-executable (GNU note, picked up by linker).
section .note.GNU-stack noalloc noexec nowrite progbits
