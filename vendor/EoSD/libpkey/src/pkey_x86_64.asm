; pkey_x86_64.asm - NASM helpers for libpkey.
;
; System V AMD64 ABI, elf64 object format. Three routines:
;
;   uint32_t pkey_rdpkru(void)
;       Returns: EAX = current PKRU value (32 bits).
;       Clobbers: EAX, ECX, EDX (all caller-saved).
;       RDPKRU (0F 01 EE) reads PKRU into EAX. ECX and EDX must be 0
;       per the Intel SDM (future-extension bits; non-zero -> #GP).
;       ~20-cycle user-mode instruction, gated by CR4.PKE (OSPKE).
;
;   void pkey_wrpkru(uint32_t pkru)
;       Inputs : EDI = new PKRU value (SysV arg 0).
;       Clobbers: EAX, ECX, EDX.
;       WRPKRU (0F 01 EF) writes EAX to PKRU. ECX and EDX must be 0.
;       ~20-cycle user-mode instruction, gated by CR4.PKE (OSPKE).
;       PKRU is per-thread; the kernel saves/restores it across context
;       switches and on kernel entry (to prevent the kernel from being
;       restricted by userspace AD/WD bits).
;
;   void pkey_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
;       Inputs : EDI = leaf, ESI = subleaf, RDX = out (uint32_t[4]).
;       Returns: out[0]=EAX, out[1]=EBX, out[2]=ECX, out[3]=EDX.
;       Used by pkey_available() to read CPUID 7:0 and test ECX[4] (OSPKE).
;
; Encoding notes: NASM 2.16.01 recognizes the `rdpkru` and `wrpkru`
; mnemonics directly. The byte sequences are 0F 01 EE and 0F 01 EF
; respectively (verified via objdump on the assembled object). The
; explicit `xor ecx,ecx` / `xor edx,edx` are required by the SDM;
; the instructions do not auto-zero these registers.

default rel

section .text

global pkey_rdpkru
pkey_rdpkru:
    xor     ecx, ecx            ; ECX = 0 (required by SDM)
    xor     edx, edx            ; EDX = 0 (required by SDM)
    rdpkru                      ; 0F 01 EE; EAX = PKRU
    ret

global pkey_wrpkru
pkey_wrpkru:
    ; in:   edi = new PKRU value (SysV arg 0)
    mov     eax, edi            ; EAX = new PKRU value (WRPKRU source)
    xor     ecx, ecx            ; ECX = 0 (required by SDM)
    xor     edx, edx            ; EDX = 0 (required by SDM)
    wrpkru                      ; 0F 01 EF; PKRU = EAX
    ret

global pkey_cpuid
pkey_cpuid:
    ; in:   edi = leaf, esi = subleaf, rdx = out (uint32_t[4])
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
