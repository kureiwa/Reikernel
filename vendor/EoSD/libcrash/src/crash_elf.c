/*
 * crash_elf.c: minimal ELF core writer for the v0.2/v0.3 crash handler.
 *
 * Async-signal-safety audit (only the symbols actually referenced on
 * this path):
 *   - getpid, getppid, getpgrp, getsid: all listed in signal-safety(7).
 *     Used to populate the prstatus pid/ppid/pgrp/sid fields.
 *   - direct loads from ucontext->uc_mcontext.gregs and stores into
 *     the caller's buffer: safe.
 *   - crash_copy_4k: NASM leaf, no libc.
 * No malloc, no memcpy, no memset, no printf, no fopen, no pthread_*,
 * no write(2) (the caller in crash.c issues the single write for the
 * whole buffer -- ELF core + user-blob section -- so the ELF builder
 * only does memory stores). Every byte of every ELF structure is
 * written via an explicit field assignment (no zero-init of the buffer
 * is assumed, since memset is not on the POSIX async-signal-safe
 * list).
 *
 * File layout (4628 bytes total):
 *   offset    0: Elf64_Ehdr            (64)
 *   offset   64: Elf64_Phdr[0] PT_NOTE (56)
 *   offset  120: Elf64_Phdr[1] PT_LOAD (56)
 *   offset  176: PT_NOTE payload:
 *                  Elf64_Nhdr           (12)
 *                  "CORE\0" + 3 pad     (8)
 *                  struct elf_prstatus  (336)
 *                                              = 356
 *   offset  532: PT_LOAD payload: 4 KB stack snapshot (4096)
 *   offset 4628: end
 *
 * The PT_LOAD segment covers [RSP, RSP+4096) at vaddr=RSP. p_align=1
 * (no alignment requirement) so the snapshot can live at any file
 * offset; gdb accepts p_align=1 for core PT_LOAD segments.
 *
 * If RSP is NULL or non-canonical, p_filesz and p_memsz are set to 0
 * and the 4 KB at offset 532 is whatever the caller's buffer happened
 * to hold (gdb will not read them, since PT_LOAD.filesz == 0). This
 * mirrors the v0.1 custom-format behavior of leaving stack_snapshot
 * untouched in the same case.
 */

#include <crash.h>
#include "crash_internal.h"

#include <elf.h>
#include <stdint.h>
#include <sys/procfs.h>
#include <sys/ucontext.h>
#include <unistd.h>

/* ASM helper from crash_x86_64.asm (also referenced by crash.c). */
extern void crash_copy_4k(void *dst, const void *src);

/* User-space address ceiling on x86-64 Linux: 47-bit (2^47). Same
 * constant as crash.c; duplicated so crash_elf.c does not need to
 * reach into crash.c's private macros. */
#define CRASH_USER_ADDR_CEILING 0x0000800000000000ULL

/* File-layout offsets, derived from struct sizes. */
#define OFF_EHDR        0u
#define OFF_PHDR_NOTE   (OFF_EHDR   + sizeof(Elf64_Ehdr))
#define OFF_PHDR_LOAD   (OFF_PHDR_NOTE + sizeof(Elf64_Phdr))
#define OFF_NOTE        (OFF_PHDR_LOAD + sizeof(Elf64_Phdr))
#define OFF_NOTE_NAME   (OFF_NOTE + sizeof(Elf64_Nhdr))
#define OFF_NOTE_DESC   (OFF_NOTE_NAME + 8)  /* "CORE\0" padded to 4-byte align */
#define OFF_STACK       (OFF_NOTE_DESC + sizeof(struct elf_prstatus))

/* Compile-time sanity check: the layout sums to CRASH_ELF_CORE_SIZE.
 * sizeof() in an integer constant expression is allowed by C11
 * 6.6.3, so _Static_assert can evaluate it. */
_Static_assert((OFF_STACK + 4096u) == CRASH_ELF_CORE_SIZE,
               "crash_elf.c: ELF core layout size mismatch");

ssize_t crash_elf_build(void *buf, size_t buf_size,
                        int sig, const siginfo_t *info,
                        const ucontext_t *uc)
{
    if (buf_size < CRASH_ELF_CORE_SIZE) {
        return -1;
    }

    uint8_t *p = (uint8_t *)buf;

    /* ---- Elf64_Ehdr ---- */
    Elf64_Ehdr *eh = (Elf64_Ehdr *)(p + OFF_EHDR);
    eh->e_ident[EI_MAG0]       = ELFMAG0;       /* 0x7f */
    eh->e_ident[EI_MAG1]       = ELFMAG1;       /* 'E'  */
    eh->e_ident[EI_MAG2]       = ELFMAG2;       /* 'L'  */
    eh->e_ident[EI_MAG3]       = ELFMAG3;       /* 'F'  */
    eh->e_ident[EI_CLASS]      = ELFCLASS64;
    eh->e_ident[EI_DATA]       = ELFDATA2LSB;
    eh->e_ident[EI_VERSION]    = EV_CURRENT;
    eh->e_ident[EI_OSABI]      = ELFOSABI_NONE;
    eh->e_ident[EI_ABIVERSION] = 0;
    /* The 7 trailing pad bytes of e_ident (indices 9..15) are
     * required to be zero by the ELF spec; write each explicitly
     * rather than memset. */
    eh->e_ident[ 9] = 0;
    eh->e_ident[10] = 0;
    eh->e_ident[11] = 0;
    eh->e_ident[12] = 0;
    eh->e_ident[13] = 0;
    eh->e_ident[14] = 0;
    eh->e_ident[15] = 0;
    eh->e_type      = ET_CORE;
    eh->e_machine   = EM_X86_64;
    eh->e_version   = EV_CURRENT;
    eh->e_entry     = 0;
    eh->e_phoff     = (Elf64_Off)OFF_PHDR_NOTE;
    eh->e_shoff     = 0;
    eh->e_flags     = 0;
    eh->e_ehsize    = (Elf64_Half)sizeof(Elf64_Ehdr);
    eh->e_phentsize = (Elf64_Half)sizeof(Elf64_Phdr);
    eh->e_phnum     = 2;
    eh->e_shentsize = 0;
    eh->e_shnum     = 0;
    eh->e_shstrndx  = 0;

    /* ---- Elf64_Phdr[0] PT_NOTE ---- */
    Elf64_Phdr *ph_note = (Elf64_Phdr *)(p + OFF_PHDR_NOTE);
    ph_note->p_type   = PT_NOTE;
    ph_note->p_flags  = PF_R;
    ph_note->p_offset = (Elf64_Off)OFF_NOTE;
    ph_note->p_vaddr  = 0;
    ph_note->p_paddr  = 0;
    ph_note->p_filesz = (Elf64_Xword)(sizeof(Elf64_Nhdr) + 8 +
                                      sizeof(struct elf_prstatus));
    ph_note->p_memsz  = 0;
    ph_note->p_align  = 4;

    /* ---- Elf64_Phdr[1] PT_LOAD (4 KB stack window at RSP) ---- */
    Elf64_Phdr *ph_load = (Elf64_Phdr *)(p + OFF_PHDR_LOAD);
    ph_load->p_type   = PT_LOAD;
    ph_load->p_flags  = PF_R;
    ph_load->p_offset = (Elf64_Off)OFF_STACK;
    ph_load->p_vaddr  = 0;       /* patched once RSP is read */
    ph_load->p_paddr  = 0;
    ph_load->p_filesz = 4096;
    ph_load->p_memsz  = 4096;
    ph_load->p_align  = 1;

    /* ---- Elf64_Nhdr ---- */
    Elf64_Nhdr *nh = (Elf64_Nhdr *)(p + OFF_NOTE);
    nh->n_namesz = 5;            /* "CORE\0" */
    nh->n_descsz = (Elf64_Word)sizeof(struct elf_prstatus);
    nh->n_type   = NT_PRSTATUS;

    /* ---- Note name "CORE\0" + 3 bytes of explicit zero pad ---- */
    uint8_t *name = p + OFF_NOTE_NAME;
    name[0] = (uint8_t)'C';
    name[1] = (uint8_t)'O';
    name[2] = (uint8_t)'R';
    name[3] = (uint8_t)'E';
    name[4] = 0;
    name[5] = 0;
    name[6] = 0;
    name[7] = 0;

    /* ---- struct elf_prstatus ---- */
    struct elf_prstatus *pr = (struct elf_prstatus *)(p + OFF_NOTE_DESC);

    /* Zero every field explicitly; the buffer is caller-owned and
     * may contain anything. memset is not async-signal-safe. */
    pr->pr_info.si_signo = 0;
    pr->pr_info.si_code  = 0;
    pr->pr_info.si_errno = 0;
    pr->pr_cursig  = 0;
    pr->pr_sigpend = 0;
    pr->pr_sighold = 0;
    pr->pr_pid     = 0;
    pr->pr_ppid    = 0;
    pr->pr_pgrp    = 0;
    pr->pr_sid     = 0;
    pr->pr_utime.tv_sec  = 0;
    pr->pr_utime.tv_usec = 0;
    pr->pr_stime.tv_sec  = 0;
    pr->pr_stime.tv_usec = 0;
    pr->pr_cutime.tv_sec  = 0;
    pr->pr_cutime.tv_usec = 0;
    pr->pr_cstime.tv_sec  = 0;
    pr->pr_cstime.tv_usec = 0;
    pr->pr_fpvalid = 0;

    /* pr_info / pr_cursig from siginfo (or fall back to `sig`). */
    if (info != NULL) {
        pr->pr_info.si_signo = info->si_signo;
        pr->pr_info.si_code  = info->si_code;
        pr->pr_info.si_errno = info->si_errno;
    } else {
        pr->pr_info.si_signo = sig;
        pr->pr_info.si_code  = 0;
        pr->pr_info.si_errno = 0;
    }
    pr->pr_cursig = (short int)sig;

    /* pid/ppid/pgrp/sid -- all async-signal-safe per POSIX.1-2008. */
    pr->pr_pid  = (int)getpid();
    pr->pr_ppid = (int)getppid();
    pr->pr_pgrp = (int)getpgrp();
    pr->pr_sid  = (int)getsid(0);

    /* pr_reg: 27 Elf64 native-word slots laid out as
     * `struct user_regs_struct` (R15, R14, ..., R8, RAX, RCX, RDX,
     * RSI, RDI, ORIG_RAX, RIP, CS, EFLAGS, RSP, SS, FS_BASE,
     * GS_BASE, DS, ES, FS, GS).
     *
     * The mcontext gregs[] array uses a different ordering
     * (REG_R8=0, REG_R9=1, ..., REG_CR2=22). The mapping below is
     * the kernel's genregs_get() translation in reverse. */
    if (uc != NULL) {
        const greg_t *g = uc->uc_mcontext.gregs;
        elf_greg_t   *r = pr->pr_reg;
        r[0]  = (elf_greg_t)g[REG_R15];
        r[1]  = (elf_greg_t)g[REG_R14];
        r[2]  = (elf_greg_t)g[REG_R13];
        r[3]  = (elf_greg_t)g[REG_R12];
        r[4]  = (elf_greg_t)g[REG_RBP];
        r[5]  = (elf_greg_t)g[REG_RBX];
        r[6]  = (elf_greg_t)g[REG_R11];
        r[7]  = (elf_greg_t)g[REG_R10];
        r[8]  = (elf_greg_t)g[REG_R9];
        r[9]  = (elf_greg_t)g[REG_R8];
        r[10] = (elf_greg_t)g[REG_RAX];
        r[11] = (elf_greg_t)g[REG_RCX];
        r[12] = (elf_greg_t)g[REG_RDX];
        r[13] = (elf_greg_t)g[REG_RSI];
        r[14] = (elf_greg_t)g[REG_RDI];
        r[15] = 0;                                          /* ORIG_RAX */
        r[16] = (elf_greg_t)g[REG_RIP];
        r[17] = (elf_greg_t)(g[REG_CSGSFS] & 0xFFFF);       /* CS */
        r[18] = (elf_greg_t)g[REG_EFL];
        r[19] = (elf_greg_t)g[REG_RSP];
        r[20] = 0;                                          /* SS */
        r[21] = 0;                                          /* FS_BASE */
        r[22] = 0;                                          /* GS_BASE */
        r[23] = 0;                                          /* DS */
        r[24] = 0;                                          /* ES */
        r[25] = (elf_greg_t)((g[REG_CSGSFS] >> 32) & 0xFFFF);  /* FS */
        r[26] = (elf_greg_t)((g[REG_CSGSFS] >> 16) & 0xFFFF);  /* GS */
    }

    /* ---- PT_LOAD payload: 4 KB stack snapshot from [RSP, RSP+4096) ---- */
    uint64_t rsp = (uc != NULL)
        ? (uint64_t)uc->uc_mcontext.gregs[REG_RSP]
        : 0;
    if (rsp != 0 && rsp < CRASH_USER_ADDR_CEILING) {
        crash_copy_4k(p + OFF_STACK, (const void *)rsp);
        ph_load->p_vaddr = (Elf64_Addr)rsp;
        ph_load->p_filesz = 4096;
        ph_load->p_memsz  = 4096;
    } else {
        /* RSP is garbage: emit an empty PT_LOAD so the program
         * header is still well-formed. The 4 KB at OFF_STACK are
         * left untouched (gdb will not read them; p_filesz == 0). */
        ph_load->p_vaddr  = 0;
        ph_load->p_filesz = 0;
        ph_load->p_memsz  = 0;
    }

    /* The caller (crash.c's crash_write_dump) appends the user-blob
     * section after the ELF core and issues a single write(2) for the
     * whole buffer (ELF core + blob section). Returning the built
     * size here so the caller knows where to append. */
    return (ssize_t)CRASH_ELF_CORE_SIZE;
}
