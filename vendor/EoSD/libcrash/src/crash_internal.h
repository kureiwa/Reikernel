#ifndef CRASH_INTERNAL_H
#define CRASH_INTERNAL_H

/*
 * Internal interface between crash.c and crash_elf.c. Not installed,
 * not caller-facing.
 */

#include <stddef.h>
#include <sys/types.h>
#include <signal.h>
#include <ucontext.h>

/*
 * Total size of the minimal ELF core file produced by crash_elf_write.
 * Used by the caller to validate buffer capacity without re-deriving
 * the layout.
 *
 *   64  Elf64_Ehdr
 *   56  Elf64_Phdr[0] PT_NOTE
 *   56  Elf64_Phdr[1] PT_LOAD
 *  356  PT_NOTE data: Elf64_Nhdr (12) + "CORE\0"+3 pad (8)
 *                       + struct elf_prstatus (336)
 * 4096  PT_LOAD data: 4 KB stack snapshot
 * ----
 * 4628  total
 */
#define CRASH_ELF_CORE_SIZE 4628u

/*
 * Build a minimal ELF core file in `buf` from `sig`/`info`/`uc`. Does
 * NOT write to fd -- the caller (crash.c's crash_write_dump) appends the
 * user-blob section after the ELF core and issues a single write(2) for
 * the whole buffer. Async-signal-safe: no malloc, no memcpy/memset, only
 * direct stores + getpid/getppid/getpgrp/getsid (all in
 * signal-safety(7)).
 *
 * The prstatus pr_reg GP-register set is the only register state written
 * into the core: the v0.2 YMM capture lives in the custom-format path
 * (crash.c's crash_capture_ymm_from_fpstate, which parses the XSAVE
 * area). The ELF core does not carry an NT_X86_XSTATE note; gdb reads
 * only the GP set from this core.
 *
 * Return:
 *   >= 0 : number of bytes built (always CRASH_ELF_CORE_SIZE on success).
 *   -1   : buf_size < CRASH_ELF_CORE_SIZE (caller falls back to the
 *          custom format).
 */
ssize_t crash_elf_build(void *buf, size_t buf_size,
                        int sig, const siginfo_t *info,
                        const ucontext_t *uc);

#endif /* CRASH_INTERNAL_H */
