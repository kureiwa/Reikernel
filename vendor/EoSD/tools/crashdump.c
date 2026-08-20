/*
 * crashdump: standalone decoder for libcrash dump files.
 *
 * Supports both the v0.1 custom binary format (crash_dump_t, ~4.7 KiB)
 * and the v0.2 minimal ELF core format (~4.6 KiB). Reads the dump
 * directly by offset -- does NOT link against libcrash, does NOT include
 * libcrash headers. The struct layout below is a byte-exact local copy
 * of crash_dump_t in libcrash/include/crash.h.
 *
 * Usage:
 *   crashdump [--json|--text|--hex] <dump-file>
 *
 * Defaults to --text. --hex prints a classic offset/hex/ASCII dump of
 * the raw file. Exit codes: 0 success, 1 usage error, 2 file/parse
 * error.
 *
 * Build: see Makefile. No external dependencies beyond libc + <elf.h>.
 */

#include <ctype.h>
#include <elf.h>
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Constants copied from libcrash (kept local to avoid coupling) ---- */

/* ASCII "CRASHDUM" as a big-endian u64: 0x43 0x52 0x41 0x53 0x48 0x44
 * 0x55 0x4D. Mirrors CRASH_MAGIC in libcrash/include/crash.h. */
#define CRASH_MAGIC 0x435241534844554DULL

/* Local copy of the v0.1 custom-format struct. Field order, types, and
 * padding match crash_dump_t exactly (verified by offset probe). */
typedef struct {
    uint64_t magic;             /* offset  0 */
    uint64_t timestamp_ns;      /* offset  8: raw TSC from rdtsc */
    int      signal_number;     /* offset 16 */
    int      _pad0;             /* offset 20: keep gpr 8-aligned */
    uint64_t gpr[16];           /* offset 24: RAX,RBX,RCX,RDX,RSI,RDI,
                                 *           RBP,RSP,R8..R15 */
    uint64_t rip;               /* offset 152 */
    uint64_t rsp;               /* offset 160 */
    uint64_t eflags;            /* offset 168 */
    uint8_t  ymm[16][32];       /* offset 176: low 256 bits of YMM0..15 */
    uint8_t  stack_snapshot[4096]; /* offset 688 */
} crash_dump_t;                 /* sizeof == 4784 */

/* GPR names in the custom-format gpr[] order (matches crash.h). */
static const char *const GPR_NAMES[16] = {
    "RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "RBP", "RSP",
    "R8",  "R9",  "R10", "R11", "R12", "R13", "R14", "R15",
};

/* Bytes of the 4 KiB stack snapshot to render in text/JSON output. */
#define STACK_PRINT_BYTES 256u

/* v0.2 user-blob extension. After the fixed crash_dump_t, trailing
 * bytes are interpreted as a sequence of records:
 *   { uint32_t key_len; uint32_t data_len; uint8_t key[key_len];
 *     uint8_t data[data_len]; }
 * No length prefix for the record stream itself; the parser walks until
 * the buffer ends. If the trailing bytes do not form a well-shaped
 * record stream, parsing stops at the first malformed record. */
#define MAX_BLOBS 64u
typedef struct {
    uint32_t         key_len;
    uint32_t         data_len;
    const uint8_t   *key;
    const uint8_t   *data;
} user_blob_t;

/* Decoded dump: a single intermediate shape produced by both the
 * custom-format and ELF-format parsers, consumed by all three
 * printers (text, JSON, hex). */
typedef struct {
    int      format;          /* 0 = custom, 1 = ELF */
    uint64_t magic;           /* CRASH_MAGIC for custom, 0 for ELF */
    uint64_t timestamp_ns;    /* custom only; 0 for ELF */
    int      signal_number;
    uint64_t gpr[16];         /* normalized to custom gpr[] order */
    uint64_t rip;
    uint64_t rsp;
    uint64_t eflags;
    uint8_t  ymm[16][32];     /* zeroed for ELF (no NT_X86_XSTATE) */
    int      ymm_present;     /* 1 for custom, 0 for ELF */
    const uint8_t *stack;     /* points into the raw file buffer */
    size_t         stack_len;
    uint64_t       stack_addr;/* vaddr == RSP for the captured window */
    user_blob_t blobs[MAX_BLOBS];
    size_t      n_blobs;
} decoded_t;

/* ---- Generic helpers ---- */

static const char *signal_name(int sig)
{
    switch (sig) {
    case SIGHUP:  return "SIGHUP";
    case SIGINT:  return "SIGINT";
    case SIGQUIT: return "SIGQUIT";
    case SIGILL:  return "SIGILL";
    case SIGTRAP: return "SIGTRAP";
    case SIGABRT: return "SIGABRT";
    case SIGBUS:  return "SIGBUS";
    case SIGFPE:  return "SIGFPE";
    case SIGKILL: return "SIGKILL";
    case SIGUSR1: return "SIGUSR1";
    case SIGSEGV: return "SIGSEGV";
    case SIGUSR2: return "SIGUSR2";
    case SIGPIPE: return "SIGPIPE";
    case SIGALRM: return "SIGALRM";
    case SIGTERM: return "SIGTERM";
    case SIGCHLD: return "SIGCHLD";
    case SIGCONT: return "SIGCONT";
    case SIGSTOP: return "SIGSTOP";
    case SIGTSTP: return "SIGTSTP";
    case SIGXCPU: return "SIGXCPU";
    case SIGXFSZ: return "SIGXFSZ";
    case SIGVTALRM: return "SIGVTALRM";
    case SIGPROF: return "SIGPROF";
    case SIGWINCH: return "SIGWINCH";
    case SIGSYS:  return "SIGSYS";
    default:      return NULL;
    }
}

static void hex_string(char *dst, const uint8_t *src, size_t n)
{
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        dst[2 * i]     = hex[src[i] >> 4];
        dst[2 * i + 1] = hex[src[i] & 0x0F];
    }
    dst[2 * n] = '\0';
}

/* ---- File loading ---- */

static int load_file(const char *path, uint8_t **out_buf, size_t *out_len)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "crashdump: cannot open %s: %s\n",
                path, strerror(errno));
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "crashdump: fseek failed on %s: %s\n",
                path, strerror(errno));
        fclose(fp);
        return -1;
    }
    long sz = ftell(fp);
    if (sz < 0) {
        fprintf(stderr, "crashdump: ftell failed on %s: %s\n",
                path, strerror(errno));
        fclose(fp);
        return -1;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "crashdump: fseek(0) failed on %s: %s\n",
                path, strerror(errno));
        fclose(fp);
        return -1;
    }
    uint8_t *buf = malloc((size_t)sz ? (size_t)sz : 1);
    if (buf == NULL) {
        fprintf(stderr, "crashdump: out of memory (%ld bytes)\n", sz);
        fclose(fp);
        return -1;
    }
    size_t got = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (got != (size_t)sz) {
        fprintf(stderr, "crashdump: short read on %s: %zu/%ld\n",
                path, got, sz);
        free(buf);
        return -1;
    }
    *out_buf = buf;
    *out_len = (size_t)sz;
    return 0;
}

/* ---- Custom-format parser ---- */

static int parse_custom(const uint8_t *buf, size_t len, decoded_t *d)
{
    if (len < sizeof(crash_dump_t)) {
        fprintf(stderr,
                "crashdump: custom dump truncated: %zu bytes (need %zu)\n",
                len, sizeof(crash_dump_t));
        return -1;
    }

    const crash_dump_t *dump = (const crash_dump_t *)buf;

    d->format        = 0;
    d->magic         = dump->magic;
    d->timestamp_ns  = dump->timestamp_ns;
    d->signal_number = dump->signal_number;
    for (int i = 0; i < 16; i++) {
        d->gpr[i] = dump->gpr[i];
    }
    d->rip    = dump->rip;
    d->rsp    = dump->rsp;
    d->eflags = dump->eflags;
    memcpy(d->ymm, dump->ymm, sizeof(d->ymm));
    d->ymm_present = 1;

    d->stack      = dump->stack_snapshot;
    d->stack_len  = sizeof(dump->stack_snapshot);
    d->stack_addr = dump->rsp;

    /* v0.2 user-blob extension: trailing bytes after the fixed struct. */
    size_t off = sizeof(crash_dump_t);
    d->n_blobs = 0;
    while (off + 8 <= len && d->n_blobs < MAX_BLOBS) {
        uint32_t key_len  = (uint32_t)buf[off]
                          | ((uint32_t)buf[off + 1] << 8)
                          | ((uint32_t)buf[off + 2] << 16)
                          | ((uint32_t)buf[off + 3] << 24);
        uint32_t data_len = (uint32_t)buf[off + 4]
                          | ((uint32_t)buf[off + 5] << 8)
                          | ((uint32_t)buf[off + 6] << 16)
                          | ((uint32_t)buf[off + 7] << 24);
        size_t need = 8u + (size_t)key_len + (size_t)data_len;
        if (need > len - off) {
            /* Malformed or truncated trailer; stop here. */
            break;
        }
        d->blobs[d->n_blobs].key_len  = key_len;
        d->blobs[d->n_blobs].data_len = data_len;
        d->blobs[d->n_blobs].key      = buf + off + 8;
        d->blobs[d->n_blobs].data     = buf + off + 8 + key_len;
        d->n_blobs++;
        off += need;
    }
    return 0;
}

/* ---- ELF core parser ----
 *
 * The libcrash ELF core layout (from src/crash_elf.c):
 *   offset    0: Elf64_Ehdr            (64)
 *   offset   64: Elf64_Phdr[0] PT_NOTE (56)
 *   offset  120: Elf64_Phdr[1] PT_LOAD (56)
 *   offset  176: PT_NOTE payload:
 *                  Elf64_Nhdr           (12)
 *                  "CORE\0" + 3 pad     (8)
 *                  struct elf_prstatus  (336)
 *   offset  532: PT_LOAD payload (4 KB stack snapshot)
 *
 * prstatus.pr_reg is 27 Elf64_Word entries at byte offset 112 inside
 * struct elf_prstatus (verified by offset probe). The kernel's
 * genregs_get() ordering is:
 *   [0]=R15 [1]=R14 [2]=R13 [3]=R12 [4]=RBP  [5]=RBX  [6]=R11 [7]=R10
 *   [8]=R9  [9]=R8  [10]=RAX [11]=RCX [12]=RDX [13]=RSI [14]=RDI
 *   [15]=ORIG_RAX [16]=RIP [17]=CS [18]=EFLAGS [19]=RSP [20]=SS
 *   [21]=FS_BASE [22]=GS_BASE [23]=DS [24]=ES [25]=FS [26]=GS
 */

/* Offset of pr_reg inside struct elf_prstatus on x86-64 glibc. */
#define PRSTATUS_PR_REG_OFF 112u
#define PRSTATUS_TOTAL_SIZE 336u

/* ELF magic: \x7f E L F. */
static int is_elf_magic(const uint8_t *buf, size_t len)
{
    return len >= 4 && buf[0] == 0x7f && buf[1] == 'E'
        && buf[2] == 'L' && buf[3] == 'F';
}

static uint64_t rd_u64_le(const uint8_t *p)
{
    return  (uint64_t)p[0]
         | ((uint64_t)p[1] << 8)
         | ((uint64_t)p[2] << 16)
         | ((uint64_t)p[3] << 24)
         | ((uint64_t)p[4] << 32)
         | ((uint64_t)p[5] << 40)
         | ((uint64_t)p[6] << 48)
         | ((uint64_t)p[7] << 56);
}

static int parse_elf(const uint8_t *buf, size_t len, decoded_t *d)
{
    if (len < sizeof(Elf64_Ehdr)) {
        fprintf(stderr, "crashdump: ELF dump truncated: %zu bytes\n", len);
        return -1;
    }
    if (buf[EI_CLASS] != ELFCLASS64) {
        fprintf(stderr, "crashdump: only ELFCLASS64 supported\n");
        return -1;
    }
    if (buf[EI_DATA] != ELFDATA2LSB) {
        fprintf(stderr, "crashdump: only little-endian ELF supported\n");
        return -1;
    }

    Elf64_Ehdr eh;
    memcpy(&eh, buf, sizeof(eh));

    /* Walk program headers, find PT_NOTE with NT_PRSTATUS, and PT_LOAD
     * for the stack window. */
    Elf64_Phdr ph_note;
    int have_note = 0;
    Elf64_Phdr ph_load;
    int have_load = 0;
    if (eh.e_phoff == 0 || eh.e_phnum == 0) {
        fprintf(stderr, "crashdump: ELF core has no program headers\n");
        return -1;
    }
    if (eh.e_phentsize < (Elf64_Half)sizeof(Elf64_Phdr)) {
        fprintf(stderr, "crashdump: e_phentsize=%u too small\n",
                eh.e_phentsize);
        return -1;
    }
    for (unsigned i = 0; i < eh.e_phnum; i++) {
        size_t off = (size_t)eh.e_phoff + (size_t)i * eh.e_phentsize;
        if (off + sizeof(Elf64_Phdr) > len) {
            break;
        }
        Elf64_Phdr ph;
        memcpy(&ph, buf + off, sizeof(ph));
        if (ph.p_type == PT_NOTE && !have_note) {
            ph_note   = ph;
            have_note = 1;
        } else if (ph.p_type == PT_LOAD && !have_load
                   && ph.p_filesz > 0) {
            ph_load   = ph;
            have_load = 1;
        }
    }
    if (!have_note) {
        fprintf(stderr, "crashdump: no PT_NOTE segment in ELF core\n");
        return -1;
    }

    /* Find the NT_PRSTATUS note inside the PT_NOTE segment. */
    if (ph_note.p_offset + ph_note.p_filesz > len) {
        fprintf(stderr, "crashdump: PT_NOTE out of bounds\n");
        return -1;
    }
    size_t n_off = (size_t)ph_note.p_offset;
    size_t n_end = n_off + (size_t)ph_note.p_filesz;

    const uint8_t *prstatus = NULL;
    size_t n = n_off;
    while (n + sizeof(Elf64_Nhdr) <= n_end) {
        Elf64_Nhdr nh;
        memcpy(&nh, buf + n, sizeof(nh));
        size_t name_padded = (nh.n_namesz + 3u) & ~3u;
        size_t desc_padded = (nh.n_descsz + 3u) & ~3u;
        if (n + sizeof(nh) + name_padded + desc_padded > n_end) {
            break;
        }
        if (nh.n_type == NT_PRSTATUS
            && nh.n_namesz == 5
            && memcmp(buf + n + sizeof(nh), "CORE", 4) == 0
            && nh.n_descsz >= PRSTATUS_TOTAL_SIZE) {
            prstatus = buf + n + sizeof(nh) + name_padded;
            break;
        }
        n += sizeof(nh) + name_padded + desc_padded;
    }
    if (prstatus == NULL) {
        fprintf(stderr,
                "crashdump: no NT_PRSTATUS note in PT_NOTE segment\n");
        return -1;
    }

    /* Extract signal + registers from prstatus. */
    int si_signo = 0;
    memcpy(&si_signo, prstatus, sizeof(si_signo)); /* pr_info.si_signo */

    const uint8_t *pr_reg = prstatus + PRSTATUS_PR_REG_OFF;
    uint64_t reg[27];
    for (int i = 0; i < 27; i++) {
        reg[i] = rd_u64_le(pr_reg + (size_t)i * 8u);
    }

    /* Map kernel pr_reg order into the custom-format gpr[] order. */
    uint64_t gpr[16];
    gpr[0]  = reg[10]; /* RAX */
    gpr[1]  = reg[5];  /* RBX */
    gpr[2]  = reg[11]; /* RCX */
    gpr[3]  = reg[12]; /* RDX */
    gpr[4]  = reg[13]; /* RSI */
    gpr[5]  = reg[14]; /* RDI */
    gpr[6]  = reg[4];  /* RBP */
    gpr[7]  = reg[19]; /* RSP */
    gpr[8]  = reg[9];  /* R8  */
    gpr[9]  = reg[8];  /* R9  */
    gpr[10] = reg[7];  /* R10 */
    gpr[11] = reg[6];  /* R11 */
    gpr[12] = reg[3];  /* R12 */
    gpr[13] = reg[2];  /* R13 */
    gpr[14] = reg[1];  /* R14 */
    gpr[15] = reg[0];  /* R15 */

    d->format        = 1;
    d->magic         = 0;
    d->timestamp_ns  = 0;
    d->signal_number = si_signo;
    memcpy(d->gpr, gpr, sizeof(gpr));
    d->rip    = reg[16];
    d->rsp    = reg[19];
    d->eflags = reg[18];
    memset(d->ymm, 0, sizeof(d->ymm));
    d->ymm_present = 0;
    d->n_blobs     = 0;

    /* Stack window: the first PT_LOAD with non-zero p_filesz. */
    if (have_load
        && ph_load.p_offset + ph_load.p_filesz <= len) {
        d->stack      = buf + (size_t)ph_load.p_offset;
        d->stack_len  = ph_load.p_filesz;
        d->stack_addr = ph_load.p_vaddr;
    } else {
        d->stack      = NULL;
        d->stack_len  = 0;
        d->stack_addr = 0;
    }
    return 0;
}

/* ---- Top-level decode ---- */

static int decode(const uint8_t *buf, size_t len, decoded_t *d)
{
    memset(d, 0, sizeof(*d));
    if (len >= 8 && rd_u64_le(buf) == CRASH_MAGIC) {
        return parse_custom(buf, len, d);
    }
    if (is_elf_magic(buf, len)) {
        return parse_elf(buf, len, d);
    }
    /* Print whatever leading bytes we have (clamped to len) so the
     * user can see what magic the file actually carries. */
    size_t n = len < 8 ? len : 8;
    fprintf(stderr, "crashdump: unrecognized dump format (first %zu bytes:",
            n);
    for (size_t i = 0; i < n; i++) {
        fprintf(stderr, " %02x", buf[i]);
    }
    if (n == 0) {
        fprintf(stderr, " file is empty)");
    } else {
        fprintf(stderr, ")");
    }
    fprintf(stderr, "\n");
    return -1;
}

/* ---- Text printer ---- */

static void print_text(const decoded_t *d)
{
    const char *fmt  = (d->format == 0) ? "custom v1" : "elf core";
    const char *sname = signal_name(d->signal_number);
    char sname_buf[32];
    if (sname == NULL) {
        snprintf(sname_buf, sizeof(sname_buf), "SIG%d", d->signal_number);
        sname = sname_buf;
    }

    printf("=== libcrash dump ===\n");
    printf("format: %s\n", fmt);
    if (d->magic != 0) {
        printf("magic: 0x%016" PRIx64 "\n", d->magic);
    }
    printf("timestamp: %" PRIu64 " ns (rdtsc)\n", d->timestamp_ns);
    printf("signal: %d (%s)\n", d->signal_number, sname);
    printf("\n");

    printf("Registers:\n");
    for (int i = 0; i < 16; i += 2) {
        printf("  %s: 0x%016" PRIx64 "    %s: 0x%016" PRIx64 "\n",
               GPR_NAMES[i],     d->gpr[i],
               GPR_NAMES[i + 1], d->gpr[i + 1]);
    }
    printf("  RIP: 0x%016" PRIx64 "\n", d->rip);
    printf("  RSP: 0x%016" PRIx64 "\n", d->rsp);
    printf("  EFLAGS: 0x%08" PRIx64 "\n", d->eflags);
    printf("\n");

    if (d->ymm_present) {
        printf("Vector registers (YMM0-YMM15, first 16 bytes each):\n");
        for (int i = 0; i < 16; i++) {
            printf("  YMM%d: ", i);
            for (int j = 0; j < 16; j++) {
                printf(" %02x", d->ymm[i][j]);
            }
            printf("\n");
        }
    } else {
        printf("Vector registers: not captured "
               "(ELF core does not carry NT_X86_XSTATE)\n");
    }
    printf("\n");

    if (d->stack != NULL && d->stack_len > 0) {
        size_t n = d->stack_len < STACK_PRINT_BYTES ? d->stack_len
                                                    : STACK_PRINT_BYTES;
        printf("Stack snapshot (%zu bytes at 0x%016" PRIx64 "):\n",
               n, d->stack_addr);
        for (size_t off = 0; off < n; off += 16) {
            printf("  0x%016" PRIx64 ":  ",
                   d->stack_addr + (uint64_t)off);
            size_t row = (n - off < 16) ? (n - off) : 16;
            for (size_t j = 0; j < 16; j++) {
                if (j < row) {
                    printf("%02x", d->stack[off + j]);
                } else {
                    fputs("  ", stdout);
                }
                if (j < 15) {
                    putchar(' ');
                    if (j == 7) {
                        putchar(' ');
                    }
                }
            }
            printf("\n");
        }
    } else {
        printf("Stack snapshot: not captured\n");
    }
    printf("\n");

    if (d->n_blobs == 0) {
        printf("User blobs: none\n");
    } else {
        printf("User blobs:\n");
        for (size_t i = 0; i < d->n_blobs; i++) {
            const user_blob_t *b = &d->blobs[i];
            printf("  [%zu] key=\"", i);
            for (uint32_t k = 0; k < b->key_len; k++) {
                uint8_t c = b->key[k];
                if (isprint(c)) {
                    putchar((char)c);
                } else {
                    printf("\\x%02x", c);
                }
            }
            printf("\" size=%u\n", b->data_len);
            size_t shown = b->data_len < 32 ? b->data_len : 32;
            printf("      ");
            for (size_t k = 0; k < shown; k++) {
                printf("%02x", b->data[k]);
                if (k == 15 && shown > 16) {
                    printf(" ");
                }
            }
            if (b->data_len > 32) {
                printf("... (%u more)", b->data_len - 32u);
            }
            printf("\n");
        }
    }
}

/* ---- JSON printer ---- */

static void json_escape_string(const uint8_t *s, uint32_t len)
{
    putchar('"');
    for (uint32_t i = 0; i < len; i++) {
        uint8_t c = s[i];
        if (c == '"' || c == '\\') {
            putchar('\\');
            putchar((char)c);
        } else if (c == '\n') {
            fputs("\\n", stdout);
        } else if (c == '\r') {
            fputs("\\r", stdout);
        } else if (c == '\t') {
            fputs("\\t", stdout);
        } else if (c < 0x20) {
            printf("\\u%04x", c);
        } else if (isprint(c)) {
            putchar((char)c);
        } else {
            printf("\\x%02x", c);
        }
    }
    putchar('"');
}

static void print_json(const decoded_t *d)
{
    const char *fmt = (d->format == 0) ? "custom" : "elf";
    const char *sname = signal_name(d->signal_number);
    char sname_buf[32];
    if (sname == NULL) {
        snprintf(sname_buf, sizeof(sname_buf), "SIG%d", d->signal_number);
        sname = sname_buf;
    }

    printf("{\n");
    printf("  \"format\": \"%s\",\n", fmt);
    if (d->magic != 0) {
        printf("  \"magic\": \"0x%016" PRIx64 "\",\n", d->magic);
    } else {
        printf("  \"magic\": null,\n");
    }
    printf("  \"timestamp_ns\": %" PRIu64 ",\n", d->timestamp_ns);
    printf("  \"signal\": %d,\n", d->signal_number);
    printf("  \"signal_name\": \"%s\",\n", sname);

    printf("  \"registers\": {\n");
    for (int i = 0; i < 16; i++) {
        printf("    \"%s\": \"0x%" PRIx64 "\"%s\n",
               GPR_NAMES[i], d->gpr[i],
               (i < 15) ? "," : "");
    }
    printf("  },\n");
    printf("  \"rip\": \"0x%" PRIx64 "\",\n", d->rip);
    printf("  \"rsp\": \"0x%" PRIx64 "\",\n", d->rsp);
    printf("  \"eflags\": \"0x%" PRIx64 "\",\n", d->eflags);

    printf("  \"ymm\": [");
    if (d->ymm_present) {
        for (int i = 0; i < 16; i++) {
            char hex[33];
            hex_string(hex, d->ymm[i], 16);
            printf("%s\"%s\"", (i == 0) ? "" : ", ", hex);
        }
    }
    printf("],\n");

    if (d->stack != NULL && d->stack_len > 0) {
        size_t n = d->stack_len < STACK_PRINT_BYTES ? d->stack_len
                                                    : STACK_PRINT_BYTES;
        char *hex = malloc(n * 2 + 1);
        if (hex != NULL) {
            hex_string(hex, d->stack, n);
            printf("  \"stack_hex\": \"%s\",\n", hex);
            printf("  \"stack_addr\": \"0x%" PRIx64 "\",\n",
                   d->stack_addr);
            free(hex);
        } else {
            printf("  \"stack_hex\": null,\n");
        }
    } else {
        printf("  \"stack_hex\": null,\n");
    }

    printf("  \"user_blobs\": [");
    for (size_t i = 0; i < d->n_blobs; i++) {
        const user_blob_t *b = &d->blobs[i];
        char *dhex = malloc((size_t)b->data_len * 2 + 1);
        if (dhex == NULL) {
            continue;
        }
        hex_string(dhex, b->data, b->data_len);
        printf("%s\n    {\"key\": ", (i == 0) ? "" : ",");
        json_escape_string(b->key, b->key_len);
        printf(", \"size\": %u, \"hex\": \"%s\"}", b->data_len, dhex);
        free(dhex);
    }
    if (d->n_blobs > 0) {
        printf("\n  ]\n");
    } else {
        printf("]\n");
    }
    printf("}\n");
}

/* ---- Raw hex printer ---- */

static void print_hex(const uint8_t *buf, size_t len)
{
    for (size_t off = 0; off < len; off += 16) {
        printf("%08zx  ", off);
        size_t row = (len - off < 16) ? (len - off) : 16;
        for (size_t j = 0; j < 16; j++) {
            if (j < row) {
                printf("%02x", buf[off + j]);
            } else {
                fputs("  ", stdout);
            }
            if (j == 7) {
                putchar(' ');
            } else {
                putchar(' ');
            }
        }
        printf(" |");
        for (size_t j = 0; j < row; j++) {
            uint8_t c = buf[off + j];
            putchar(isprint(c) ? (char)c : '.');
        }
        printf("|\n");
    }
}

/* ---- main ---- */

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [--json|--text|--hex] <dump-file>\n"
        "  Decode a libcrash dump (custom or ELF core format).\n"
        "  --text (default): human-readable plaintext.\n"
        "  --json:           machine-readable JSON.\n"
        "  --hex:            raw offset/hex/ASCII dump of the file.\n",
        prog);
}

int main(int argc, char **argv)
{
    enum { OUT_TEXT, OUT_JSON, OUT_HEX } mode = OUT_TEXT;
    const char *path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) {
            mode = OUT_JSON;
        } else if (strcmp(argv[i], "--text") == 0) {
            mode = OUT_TEXT;
        } else if (strcmp(argv[i], "--hex") == 0) {
            mode = OUT_HEX;
        } else if (strcmp(argv[i], "-h") == 0
                || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (argv[i][0] == '-' && argv[i][1] != '\0') {
            fprintf(stderr, "crashdump: unknown option %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        } else if (path == NULL) {
            path = argv[i];
        } else {
            fprintf(stderr, "crashdump: extra argument %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (path == NULL) {
        usage(argv[0]);
        return 1;
    }

    uint8_t *buf = NULL;
    size_t   len = 0;
    if (load_file(path, &buf, &len) != 0) {
        return 2;
    }

    int rc = 0;
    if (mode == OUT_HEX) {
        print_hex(buf, len);
    } else {
        decoded_t d;
        if (decode(buf, len, &d) != 0) {
            rc = 2;
        } else if (mode == OUT_JSON) {
            print_json(&d);
        } else {
            print_text(&d);
        }
    }
    free(buf);
    return rc;
}
