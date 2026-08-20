; tick_x86_64.asm - NASM helpers for libtick.
;
; System V AMD64 ABI, elf64 object format. Three routines:
;   uint64_t tick_rdtsc(void)   - LFENCE;RDTSC, returns TSC in rax.
;   uint64_t tick_rdtscp(void)  - RDTSCP;LFENCE, returns TSC in rax.
;                                  Caller must ensure RDTSCP is supported
;                                  (CPUID.80000001H:EDX[27]).
;   void tick_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
;                               - CPUID with leaf/subleaf, writes
;                                 EAX/EBX/ECX/EDX to out[0..3].
;
; LFENCE before RDTSC is the Intel SDM Vol 2B RDTSC recipe for ordering
; prior loads before the TSC read. RDTSCP waits for prior instructions to
; retire before reading the TSC but does not prevent later instructions
; from being reordered ahead of it; LFENCE after RDTSCP closes that gap,
; matching the kernel's rdtsc_ordered() alternative sequence in
; arch/x86/include/asm/tsc.h.

default rel

section .text

global tick_rdtsc
tick_rdtsc:
    lfence
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    ret

global tick_rdtscp
tick_rdtscp:
    rdtscp
    lfence
    shl     rdx, 32
    or      rax, rdx
    ret

global tick_cpuid
tick_cpuid:
    ; in:   edi = leaf, esi = subleaf, rdx = out (pointer to uint32_t[4])
    ; out:  out[0]=eax, out[1]=ebx, out[2]=ecx, out[3]=edx
    ; CPUID clobbers EAX/EBX/ECX/EDX. RDX holds the out pointer, so save it
    ; in R8 (caller-saved scratch) before CPUID. RBX is callee-saved; push it.
    push    rbx
    mov     eax, edi            ; leaf
    mov     ecx, esi            ; subleaf
    mov     r8,  rdx            ; preserve out pointer
    cpuid
    mov     [r8 +  0], eax      ; out[0]
    mov     [r8 +  4], ebx      ; out[1]
    mov     [r8 +  8], ecx      ; out[2]
    mov     [r8 + 12], edx      ; out[3]
    pop     rbx
    ret

; Mark stack as non-executable (GNU note, picked up by linker).
section .note.GNU-stack noalloc noexec nowrite progbits
