#!/bin/sh
# test_crashdump.sh: produce libcrash dump files and run crashdump on them.
#
# For each format (custom, ELF):
#   1. Best-effort: generate the dump via libcrash (if it builds and
#      produces a valid dump).
#   2. Fallback: if libcrash is unavailable or produced an invalid dump,
#      generate a byte-identical dump with a standalone generator that
#      does not depend on libcrash.
#
# Then run crashdump on each dump in --text, --json, and --hex modes
# and verify the output contains the expected fields.
#
# Usage: test_crashdump.sh <libcrash-dir> <crashdump-binary>
# Exit codes: 0 pass, 1 fail.

set -u

LIBCRASH_DIR="${1:-../libcrash}"
CRASHDUMP="${2:-./crashdump}"
TMPDIR_TEST="/tmp/crashdump_test_$$"

fail() {
    echo "FAIL: $*" >&2
    rm -rf "$TMPDIR_TEST"
    exit 1
}

mkdir -p "$TMPDIR_TEST" || fail "mkdir"

CUSTOM_DUMP="$TMPDIR_TEST/custom.dump"
ELF_DUMP="$TMPDIR_TEST/elf.dump"

# ---- 1. Always build the standalone generator (no libcrash dep). ----
GEN_SA="$TMPDIR_TEST/gen_standalone"
cat > "$TMPDIR_TEST/gen_standalone.c" <<'EOF'
/* gen_standalone: synthesize a libcrash dump file without linking
 * against libcrash. Uses a local byte-exact copy of crash_dump_t and
 * constructs a minimal ELF core by hand. The dump is filled with
 * deterministic, recognizable values so the test patterns can be
 * specific. */
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/procfs.h>
#include <unistd.h>

#define CRASH_MAGIC 0x435241534844554DULL
typedef struct {
    uint64_t magic;
    uint64_t timestamp_ns;
    int      signal_number;
    int      _pad0;
    uint64_t gpr[16];
    uint64_t rip;
    uint64_t rsp;
    uint64_t eflags;
    uint8_t  ymm[16][32];
    uint8_t  stack_snapshot[4096];
} crash_dump_t;

/* Offset of pr_reg inside struct elf_prstatus (x86-64 glibc). */
#define PR_REG_OFF 112u

static void write_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            perror("write");
            _exit(2);
        }
        p += w;
        n -= (size_t)w;
    }
}

static int gen_custom(const char *path) {
    crash_dump_t d;
    memset(&d, 0, sizeof(d));
    d.magic         = CRASH_MAGIC;
    d.timestamp_ns  = 0x1122334455667788ULL;
    d.signal_number = SIGSEGV;
    for (int i = 0; i < 16; i++) {
        d.gpr[i] = (uint64_t)(i + 1) * 0x11;
    }
    d.rip    = 0x0000555555555a3cULL;
    d.rsp    = 0x00007ffc12345678ULL;
    d.eflags = 0x00000046ULL;
    for (int i = 0; i < 16; i++) {
        memset(d.ymm[i], (uint8_t)(i + 1), 32);
    }
    for (int i = 0; i < 4096; i++) {
        d.stack_snapshot[i] = (uint8_t)(i & 0xff);
    }
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return 2; }
    write_all(fd, &d, sizeof(d));
    close(fd);
    return 0;
}

static int gen_elf(const char *path) {
    unsigned char core[4628];
    memset(core, 0, sizeof(core));

    Elf64_Ehdr eh;
    memset(&eh, 0, sizeof(eh));
    eh.e_ident[EI_MAG0] = ELFMAG0;
    eh.e_ident[EI_MAG1] = ELFMAG1;
    eh.e_ident[EI_MAG2] = ELFMAG2;
    eh.e_ident[EI_MAG3] = ELFMAG3;
    eh.e_ident[EI_CLASS] = ELFCLASS64;
    eh.e_ident[EI_DATA] = ELFDATA2LSB;
    eh.e_ident[EI_VERSION] = EV_CURRENT;
    eh.e_type = ET_CORE;
    eh.e_machine = EM_X86_64;
    eh.e_version = EV_CURRENT;
    eh.e_phoff = sizeof(Elf64_Ehdr);
    eh.e_ehsize = sizeof(Elf64_Ehdr);
    eh.e_phentsize = sizeof(Elf64_Phdr);
    eh.e_phnum = 2;
    memcpy(core, &eh, sizeof(eh));

    Elf64_Phdr ph_note;
    memset(&ph_note, 0, sizeof(ph_note));
    ph_note.p_type = PT_NOTE;
    ph_note.p_flags = PF_R;
    ph_note.p_offset = sizeof(Elf64_Ehdr) + 2 * sizeof(Elf64_Phdr);
    ph_note.p_filesz = sizeof(Elf64_Nhdr) + 8 + sizeof(struct elf_prstatus);
    ph_note.p_align = 4;
    memcpy(core + sizeof(Elf64_Ehdr), &ph_note, sizeof(ph_note));

    Elf64_Phdr ph_load;
    memset(&ph_load, 0, sizeof(ph_load));
    ph_load.p_type = PT_LOAD;
    ph_load.p_flags = PF_R;
    ph_load.p_offset = sizeof(Elf64_Ehdr) + 2 * sizeof(Elf64_Phdr)
                       + sizeof(Elf64_Nhdr) + 8 + sizeof(struct elf_prstatus);
    ph_load.p_vaddr = 0x00007ffc12345678ULL;
    ph_load.p_filesz = 4096;
    ph_load.p_memsz = 4096;
    ph_load.p_align = 1;
    memcpy(core + sizeof(Elf64_Ehdr) + sizeof(Elf64_Phdr),
           &ph_load, sizeof(ph_load));

    Elf64_Nhdr nh;
    memset(&nh, 0, sizeof(nh));
    nh.n_namesz = 5;
    nh.n_descsz = sizeof(struct elf_prstatus);
    nh.n_type = NT_PRSTATUS;
    size_t note_off = sizeof(Elf64_Ehdr) + 2 * sizeof(Elf64_Phdr);
    memcpy(core + note_off, &nh, sizeof(nh));
    memcpy(core + note_off + sizeof(Elf64_Nhdr), "CORE\0\0\0\0", 8);

    size_t pr_off = note_off + sizeof(Elf64_Nhdr) + 8;
    int si_signo = SIGSEGV;
    memcpy(core + pr_off, &si_signo, sizeof(si_signo));
    uint64_t pr_reg[27];
    memset(pr_reg, 0, sizeof(pr_reg));
    pr_reg[0]  = 0x1111;
    pr_reg[10] = 0xaaaa;
    pr_reg[16] = 0x0000555555555a3cULL;
    pr_reg[18] = 0x00000046ULL;
    pr_reg[19] = 0x00007ffc12345678ULL;
    memcpy(core + pr_off + PR_REG_OFF, pr_reg, sizeof(pr_reg));

    size_t stack_off = ph_load.p_offset;
    for (int i = 0; i < 4096; i++) {
        core[stack_off + i] = (unsigned char)(i & 0xff);
    }

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return 2; }
    write_all(fd, core, sizeof(core));
    close(fd);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: gen_standalone <custom|elf> <dump-file>\n");
        return 2;
    }
    if (strcmp(argv[1], "elf") == 0) return gen_elf(argv[2]);
    return gen_custom(argv[2]);
}
EOF
CFLAGS_GEN="-std=c11 -Wall -Wextra -Werror -pedantic -O2 -D_GNU_SOURCE"
echo "==> compiling standalone generator"
if ! gcc $CFLAGS_GEN -o "$GEN_SA" "$TMPDIR_TEST/gen_standalone.c"; then
    fail "standalone generator compile failed"
fi

# ---- 2. Best-effort: build libcrash + libcrash-linked generator. ----
HAVE_LIBCRASH=0
GEN_LC=""
echo "==> building libcrash (best-effort)"
if make -C "$LIBCRASH_DIR" >/dev/null 2>"$TMPDIR_TEST/libcrash.err"; then
    GEN_LC="$TMPDIR_TEST/gen_dump"
    cat > "$TMPDIR_TEST/gen_dump.c" <<'EOF'
#include <crash.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: gen_dump <custom|elf> <dump-file>\n");
        return 2;
    }
    int use_elf = (strcmp(argv[1], "elf") == 0);
    const char *path = argv[2];

    struct rlimit rl = { 0, 0 };
    setrlimit(RLIMIT_CORE, &rl);

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return 2; }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 2; }
    if (pid == 0) {
        size_t bufsz = crash_min_buffer_size();
        char *buf = malloc(bufsz);
        if (!buf) _exit(100);
        if (use_elf) {
            if (crash_install_elf(buf, bufsz, fd, NULL, 0,
                                  CRASH_AFTER_RERAISE,
                                  CRASH_FORMAT_ELF) != CRASH_OK) {
                _exit(100);
            }
        } else {
            if (crash_install(buf, bufsz, fd, NULL, 0,
                              CRASH_AFTER_RERAISE) != CRASH_OK) {
                _exit(100);
            }
        }
        volatile int *p = NULL;
        *p = 42;
        _exit(99);
    }
    int status;
    if (waitpid(pid, &status, 0) < 0) { perror("waitpid"); return 2; }
    if (!WIFSIGNALED(status) || WTERMSIG(status) != SIGSEGV) {
        fprintf(stderr, "child did not die via SIGSEGV (status=0x%x)\n",
                status);
        return 2;
    }
    close(fd);
    return 0;
}
EOF
    if gcc $CFLAGS_GEN -I"$LIBCRASH_DIR/include" -o "$GEN_LC" \
            "$TMPDIR_TEST/gen_dump.c" "$LIBCRASH_DIR/libcrash.a" \
            2>"$TMPDIR_TEST/gen.err"; then
        HAVE_LIBCRASH=1
    else
        echo "  libcrash-linked generator compile failed; using standalone"
        sed 's/^/    /' "$TMPDIR_TEST/gen.err" | head -8
    fi
else
    echo "  libcrash build failed; using standalone generator"
    sed 's/^/    /' "$TMPDIR_TEST/libcrash.err" | head -4
fi

# ---- 3. Generate each dump, validating magic; fall back per-dump. ----
# validate_dump <path> <custom|elf>: returns 0 if the dump has the
# right magic bytes, 1 otherwise.
validate_dump() {
    local path="$1" fmt="$2"
    if [ ! -s "$path" ]; then
        return 1
    fi
    local b0 b1 b2 b3 b4 b5 b6 b7
    b0=$(od -An -tu1 -N1 "$path" | tr -d ' ')
    b1=$(od -An -tu1 -j1 -N1 "$path" | tr -d ' ')
    b2=$(od -An -tu1 -j2 -N1 "$path" | tr -d ' ')
    b3=$(od -An -tu1 -j3 -N1 "$path" | tr -d ' ')
    if [ "$fmt" = "elf" ]; then
        # \x7f E L F = 127 69 76 70
        [ "$b0" = "127" ] && [ "$b1" = "69" ] \
            && [ "$b2" = "76" ] && [ "$b3" = "70" ]
        return $?
    else
        # CRASH_MAGIC little-endian: 4D 55 44 48 53 41 52 43
        # = 77 85 68 72 83 65 82 67
        b4=$(od -An -tu1 -j4 -N1 "$path" | tr -d ' ')
        b5=$(od -An -tu1 -j5 -N1 "$path" | tr -d ' ')
        b6=$(od -An -tu1 -j6 -N1 "$path" | tr -d ' ')
        b7=$(od -An -tu1 -j7 -N1 "$path" | tr -d ' ')
        [ "$b0" = "77" ] && [ "$b1" = "85" ] && [ "$b2" = "68" ] \
            && [ "$b3" = "72" ] && [ "$b4" = "83" ] && [ "$b5" = "65" ] \
            && [ "$b6" = "82" ] && [ "$b7" = "67" ]
        return $?
    fi
}

# generate_dump <custom|elf> <out-path>
generate_dump() {
    local fmt="$1" out="$2"
    if [ "$HAVE_LIBCRASH" -eq 1 ]; then
        if "$GEN_LC" "$fmt" "$out" 2>/dev/null && validate_dump "$out" "$fmt"; then
            echo "==> generated $fmt-format dump (libcrash)"
            return 0
        fi
        echo "  libcrash $fmt dump invalid or generation failed; falling back"
    fi
    "$GEN_SA" "$fmt" "$out" || fail "standalone $fmt generation failed"
    echo "==> generated $fmt-format dump (standalone)"
}

generate_dump custom "$CUSTOM_DUMP"
generate_dump elf    "$ELF_DUMP"

# ---- 4. Run crashdump on each dump in each mode and verify. ----
PASS=0
FAIL=0

check_contains() {
    # check_contains <label> <file> <pattern...>
    # Uses grep -F (fixed-string) so JSON brackets/quotes match literally.
    local label="$1" file="$2"; shift 2
    for pat in "$@"; do
        if ! grep -qF -- "$pat" "$file"; then
            echo "  [$label] MISSING: '$pat'"
            return 1
        fi
    done
    return 0
}

run_case() {
    # run_case <label> <dump-file> <mode> <expected-pattern...>
    local label="$1" dump="$2" mode="$3"; shift 3
    local out="$TMPDIR_TEST/${label}.out"
    if ! "$CRASHDUMP" "$mode" "$dump" >"$out" 2>&1; then
        echo "  [$label] crashdump exited non-zero; output was:"
        sed 's/^/      /' "$out" | head -20
        FAIL=$((FAIL + 1))
        return
    fi
    if check_contains "$label" "$out" "$@"; then
        echo "  [$label] PASS"
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        echo "  [$label] output was:"
        sed 's/^/      /' "$out" | head -40
    fi
}

echo "==> custom --text"
run_case "custom_text" "$CUSTOM_DUMP" "--text" \
    "format: custom v1" \
    "magic: 0x435241534844554d" \
    "timestamp:" \
    "signal: 11 (SIGSEGV)" \
    "Registers:" \
    "RIP: 0x" \
    "RSP: 0x" \
    "EFLAGS: 0x" \
    "Vector registers (YMM0-YMM15" \
    "Stack snapshot (256 bytes at 0x" \
    "User blobs: none"

echo "==> custom --json"
run_case "custom_json" "$CUSTOM_DUMP" "--json" \
    '"format": "custom"' \
    '"magic": "0x435241534844554d"' \
    '"timestamp_ns":' \
    '"signal": 11,' \
    '"signal_name": "SIGSEGV"' \
    '"registers":' \
    '"rip":' \
    '"rsp":' \
    '"eflags":' \
    '"ymm":' \
    '"stack_hex":' \
    '"user_blobs": []'

echo "==> custom --hex"
# CRASH_MAGIC = 0x435241534844554D is stored little-endian on disk, so
# byte 0 is 0x4D ('M') and the ASCII rendering of the first 8 bytes is
# "MUDHSARC" (== "CRASHDUM" reversed).
run_case "custom_hex" "$CUSTOM_DUMP" "--hex" \
    "4d 55 44 48 53 41 52 43" \
    "MUDHSARC"

echo "==> elf --text"
run_case "elf_text" "$ELF_DUMP" "--text" \
    "format: elf core" \
    "timestamp: 0 ns" \
    "signal: 11 (SIGSEGV)" \
    "Registers:" \
    "RIP: 0x" \
    "RSP: 0x"

echo "==> elf --json"
run_case "elf_json" "$ELF_DUMP" "--json" \
    '"format": "elf"' \
    '"magic": null,' \
    '"timestamp_ns": 0,' \
    '"signal": 11,' \
    '"signal_name": "SIGSEGV"' \
    '"registers":'

echo "==> elf --hex"
run_case "elf_hex" "$ELF_DUMP" "--hex" \
    "7f 45 4c 46"

# ---- 5. Error paths. ----
echo "==> error: missing file"
if "$CRASHDUMP" "/nonexistent/path" >/dev/null 2>&1; then
    echo "  [err_missing] FAIL (expected non-zero exit)"
    FAIL=$((FAIL + 1))
else
    echo "  [err_missing] PASS"
    PASS=$((PASS + 1))
fi

echo "==> error: no args"
if "$CRASHDUMP" >/dev/null 2>&1; then
    echo "  [err_noargs] FAIL (expected non-zero exit)"
    FAIL=$((FAIL + 1))
else
    echo "  [err_noargs] PASS"
    PASS=$((PASS + 1))
fi

# ---- Summary. ----
echo ""
echo "==== crashdump test summary: $PASS passed, $FAIL failed ===="
if [ "$FAIL" -ne 0 ]; then
    rm -rf "$TMPDIR_TEST"
    exit 1
fi
rm -rf "$TMPDIR_TEST"
exit 0
