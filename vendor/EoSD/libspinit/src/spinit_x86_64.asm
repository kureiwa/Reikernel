; spinit_x86_64.asm - NASM helpers for libspinit.
;
; System V AMD64 ABI, elf64 object format. Two routines:
;   uint64_t spinit_rdtsc(void)       - LFENCE;RDTSC, returns TSC in rax.
;   void spinit_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
;                                     - CPUID with leaf/subleaf, writes
;                                       EAX/EBX/ECX/EDX to out[0..3].
;
; LFENCE before RDTSC is the Intel SDM Vol 2B RDTSC recipe for ordering
; prior loads/stores before the TSC read. The hot-path spin loop does
; not call spinit_rdtsc; only calibration does.

default rel

section .text

global spinit_rdtsc
spinit_rdtsc:
    lfence
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    ret

global spinit_cpuid
spinit_cpuid:
    ; in:   edi = leaf, esi = subleaf, rdx = out (pointer to uint32_t[4])
    ; out:  out[0]=eax, out[1]=ebx, out[2]=ecx, out[3]=edx
    ; CPUID writes to EAX/EBX/ECX/EDX, which on x86_64 zero-extends into
    ; RAX/RBX/RCX/RDX and so destroys the RDX pointer. Save RDX into R8
    ; (caller-saved scratch) before CPUID. RBX is callee-saved, so push it.
    push    rbx
    mov     eax, edi            ; leaf
    mov     ecx, esi            ; subleaf
    mov     r8,  rdx            ; preserve out pointer across CPUID
    cpuid
    mov     [r8 +  0], eax      ; out[0]
    mov     [r8 +  4], ebx      ; out[1]
    mov     [r8 +  8], ecx      ; out[2]
    mov     [r8 + 12], edx      ; out[3]
    pop     rbx
    ret

; Mark stack as non-executable (GNU note, picked up by linker).
section .note.GNU-stack noalloc noexec nowrite progbits
