; spoon_x86_64.asm -- asymmetric coroutine context switch for x86_64 Linux.
; System V AMD64 ABI. Assembled with: nasm -f elf64.
;
; The switch saves callee-saved GPRs (RBX, RBP, R12-R15), MXCSR control
; bits, x87 control word, RSP, and the return address (RIP) from the
; outgoing context, then loads the same set from the incoming context
; and jumps to the incoming RIP.
;
; XMM/YMM register values are caller-saved per the ABI and are NOT
; preserved. MXCSR and x87 CW ARE preserved (callee-saved per ABI
; table 3.4 + section 3.2.2).

%define OFF_SP    0
%define OFF_RBX   8
%define OFF_RBP   16
%define OFF_R12   24
%define OFF_R13   32
%define OFF_R14   40
%define OFF_R15   48
%define OFF_RIP   56
%define OFF_MXCSR 64
%define OFF_X87CW 68

extern spoon_entry

section .text

; void spoon_switch(spoon_co_t *from, spoon_co_t *to)
;   from = rdi, to = rsi
global spoon_switch
spoon_switch:
    ; --- Save outgoing context into `from` (rdi) ---
    mov     [rdi + OFF_RBX],  rbx
    mov     [rdi + OFF_RBP],  rbp
    mov     [rdi + OFF_R12],  r12
    mov     [rdi + OFF_R13],  r13
    mov     [rdi + OFF_R14],  r14
    mov     [rdi + OFF_R15],  r15
    stmxcsr [rdi + OFF_MXCSR]
    fnstcw  [rdi + OFF_X87CW]

    ; Save return address (at [rsp], pushed by the call) and old RSP.
    ; `pop rax` reads the return address into rax and advances rsp to the
    ; value the caller had before `call` pushed the return address. rsp is
    ; overwritten below by `mov rsp, [rsi + OFF_SP]` before any further
    ; stack access, so the side effect of moving rsp is invisible. This
    ; saves one instruction vs the older mov+lea+mov sequence.
    pop     rax
    mov     [rdi + OFF_RIP], rax
    mov     [rdi + OFF_SP], rsp

    ; --- Load incoming context from `to` (rsi) ---
    mov     rsp, [rsi + OFF_SP]
    mov     rbx, [rsi + OFF_RBX]
    mov     rbp, [rsi + OFF_RBP]
    mov     r12, [rsi + OFF_R12]
    mov     r13, [rsi + OFF_R13]
    mov     r14, [rsi + OFF_R14]
    mov     r15, [rsi + OFF_R15]
    ldmxcsr [rsi + OFF_MXCSR]
    fldcw   [rsi + OFF_X87CW]

    ; Jump to the incoming RIP. We use `jmp` (not `ret`) because the
    ; saved RIP is the return address of the target's original
    ; spoon_switch call. RSP is already set to the target's pre-call
    ; value, so jumping to the return address is equivalent to the
    ; original call returning. Indirect jump through memory saves one
    ; instruction vs mov rax,[mem]; jmp rax.
    jmp     [rsi + OFF_RIP]

; void spoon_trampoline(void)
;   Entered on first switch into a new coroutine. rbx = spoon_co_t*.
;   Sets up rdi (first System V arg) = rbx, calls spoon_entry.
;   spoon_entry never returns (it switches away when fn completes).
global spoon_trampoline
spoon_trampoline:
    mov     rdi, rbx                 ; first arg = co
    ; Align stack to 16 bytes before call (rsp is 16-aligned on entry
    ; since we jmp'd here with a 16-aligned sp; call will push 8 bytes,
    ; making rsp mod 16 == 8 inside spoon_entry, per the ABI).
    call    spoon_entry
    ; Should never reach here.
    ud2
