; pmu_x86_64.asm - NASM helpers for libpmu v0.2 rdpmc fast path.
;
; System V AMD64 ABI, elf64 object format. Two routines:
;
;   uint64_t pmu_rdpmc(uint32_t index)
;       Inputs : EDI = RDPMC index. For general PMCs this is
;                mmap_page->index - 1. For Intel fixed-function counters
;                the kernel sets bit 30 of mmap_page->index; subtracting 1
;                leaves ECX[30]=1 so RDPMC selects the fixed counter space
;                and ECX[0..2] picks IA32_FIXED_CTR0/1/2. No special casing
;                is needed here.
;       Returns: RAX = full 64-bit counter value (EDX:EAX concatenated).
;       Clobbers: RAX, RCX, RDX (all caller-saved).
;
;   The lfence;rdpmc;lfence sequence matches the kernel's
;   x86_perf_event_update() (arch/x86/events/core.c) and the recipe in
;   Documentation/arch/x86/ for userspace rdpmc reads via the perf_event
;   mmap page. The leading lfence orders the mmap_page->index / offset
;   loads ahead of the counter read; the trailing lfence orders the
;   counter read ahead of the mmap_page->lock re-read.
;
;   void pmu_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
;       Inputs : EDI = CPUID leaf, ESI = subleaf, RDX = out (uint32_t[4]).
;       Returns: out[0]=EAX, out[1]=EBX, out[2]=ECX, out[3]=EDX.
;       Used by pmu_open() to read CPUID.0H and compare the vendor string
;       (EBX,EDX,ECX in Intel order: "GenuineIntel" / "AuthenticAMD")
;       against the Intel fixed-counter assumption. Vendor detection is
;       informational for v0.2: the kernel's perf_event_open programs the
;       appropriate event-select MSR for the platform, so the rdpmc
;       recipe (index - 1) is identical for both vendors.

default rel

section .text

global pmu_rdpmc
pmu_rdpmc:
    mov     ecx, edi            ; index arg in EDI (SysV) -> ECX for RDPMC
    lfence
    rdpmc                       ; EDX:EAX = PMC value
    lfence
    shl     rdx, 32
    or      rax, rdx            ; RAX = (uint64_t)EDX << 32 | EAX
    ret

global pmu_cpuid
pmu_cpuid:
    ; in:   edi = leaf, esi = subleaf, rdx = out (uint32_t[4])
    ; out:  out[0]=eax, out[1]=ebx, out[2]=ecx, out[3]=edx
    ; CPUID clobbers EAX/EBX/ECX/EDX. RDX holds the out pointer; save it
    ; in R8 (caller-saved scratch) before CPUID. RBX is callee-saved, so
    ; push it.
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
