; topo_x86_64.asm - NASM helpers for libtopo.
;
; System V AMD64 ABI, elf64 object format. Three routines:
;
;   void topo_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
;       Inputs : EDI = CPUID leaf, ESI = subleaf, RDX = out (uint32_t[4]).
;       Returns: out[0]=EAX, out[1]=EBX, out[2]=ECX, out[3]=EDX.
;       Clobbers: RAX, RCX, RDX, R8 (all caller-saved). RBX is
;                 callee-saved so it is pushed/popped.
;       Same pattern as libpmu/src/pmu_x86_64.asm and
;       libspinit/src/spinit_x86_64.asm. Used by topo_probe (topology
;       enumeration via leaf 0x1F / 0xB), topo_cache_info (leaf 4),
;       and the RDPID-feature check (leaf 7).
;
;   uint32_t topo_rdpid(void)
;       Returns: EAX = IA32_TSC_AUX (low 32 bits).
;       Clobbers: RAX, RCX.
;       RDPID (encoding F3 0F C7 /F8) reads the IA32_TSC_AUX MSR into
;       ECX without the TSC side-effect of RDTSCP and without the
;       serializing overhead. On Linux the kernel writes
;       "cpu_number | (node_id << 12)" to IA32_TSC_AUX (see
;       arch/x86/kernel/tsc.c and arch/x86/kernel/cpu/common.c), so
;       the low 12 bits are the CPU number; topo_getcpu masks the
;       result with 0xFFF before returning. Caller MUST verify
;       CPUID.7.0:ECX[1] is set before invoking this routine on a
;       CPU that might not support RDPID -- executing RDPID on an
;       unsupported CPU raises #UD. topo_have_rdpid() in topo.c does
;       the check.
;
;   uint32_t topo_rdtscp_ecx(void)
;       Returns: EAX = IA32_TSC_AUX (low 32 bits). The TSC value in
;                EDX:EAX is discarded.
;       Clobbers: RAX, RCX, RDX.
;       RDTSCP reads TSC into EDX:EAX and IA32_TSC_AUX into ECX. The
;       instruction is serializing (it fences prior loads) and is
;       available on essentially every x86_64 CPU shipped since 2008
;       (Nehalem / Barcelona). libtopo does not call this on the
;       getcpu fast path (RDPID is faster and non-serializing); it
;       is exposed for callers that want the TSC_AUX value with the
;       ordering guarantee of RDTSCP (e.g. pairing with a TSC read).

default rel

section .text

global topo_cpuid
topo_cpuid:
    ; in:   edi = leaf, esi = subleaf, rdx = out (uint32_t[4])
    ; out:  out[0]=eax, out[1]=ebx, out[2]=ecx, out[3]=edx
    ; CPUID writes EAX/EBX/ECX/EDX and so destroys the RDX pointer.
    ; Save RDX into R8 (caller-saved scratch) before CPUID. RBX is
    ; callee-saved, so push it.
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

global topo_rdpid
topo_rdpid:
    ; RDPID reads IA32_TSC_AUX into the destination register. NASM
    ; requires an explicit 32/64-bit GPR operand (unlike RDTSCP, which
    ; is fixed to EDX:EAX,ECX). We pick EAX so no extra mov is needed
    ; for the SysV return-value convention.
    rdpid   eax                     ; EAX = IA32_TSC_AUX
    ret

global topo_rdtscp_ecx
topo_rdtscp_ecx:
    ; RDTSCP: EDX:EAX = TSC, ECX = IA32_TSC_AUX. Discard TSC, return
    ; ECX in EAX.
    rdtscp
    mov     eax, ecx
    ret

; Mark stack as non-executable (GNU note, picked up by linker).
section .note.GNU-stack noalloc noexec nowrite progbits
