# libcrash: async-signal-safe crash/minidump writer.

CC       = gcc
NASM    ?= nasm
AR       = ar
CFLAGS   = -std=c11 -Wall -Wextra -Werror -pedantic -O2 -D_GNU_SOURCE
INCLUDES = -Iinclude

# AVX flag for test_ymm, set by the top-level Makefile's CPU detection.
# Falls back to local detection if run standalone.
ifeq ($(HAS_AVX),)
    HAS_AVX := $(shell grep -q ' avx ' /proc/cpuinfo && echo yes)
endif

LIB        = libcrash.a
LIB_OBJS   = src/crash.o src/crash_elf.o src/crash_x86_64.o

TEST_BINS  = tests/test_basic tests/test_sigsegv tests/test_sigabrt \
	     tests/test_reentry tests/test_elf tests/test_fork \
	     tests/test_ymm tests/test_waitpid_echild tests/test_user_blob
TEST_OBJS  = $(TEST_BINS:=.o)

BENCH_BINS = bench/bench_dump bench/bench_install
BENCH_OBJS = $(BENCH_BINS:=.o)

.PHONY: all test bench clean

all: $(LIB)

$(LIB): $(LIB_OBJS)
	$(AR) rcs $@ $^

%.o: %.c include/crash.h src/crash_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

src/crash_x86_64.o: src/crash_x86_64.asm
	$(NASM) -f elf64 -o $@ $<

tests/test_basic:   tests/test_basic.o   $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lcrash

tests/test_sigsegv: tests/test_sigsegv.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lcrash

tests/test_sigabrt: tests/test_sigabrt.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lcrash

tests/test_reentry: tests/test_reentry.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lcrash

tests/test_elf:  tests/test_elf.o  $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lcrash

tests/test_fork: tests/test_fork.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lcrash

ifeq ($(HAS_AVX),yes)
tests/test_ymm.o: tests/test_ymm.c include/crash.h
	$(CC) $(CFLAGS) -mavx $(INCLUDES) -c -o $@ $<
else
tests/test_ymm.o: tests/test_ymm.c include/crash.h
	$(CC) $(CFLAGS) -DNO_AVX $(INCLUDES) -c -o $@ $<
endif

tests/test_ymm: tests/test_ymm.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lcrash

tests/test_waitpid_echild: tests/test_waitpid_echild.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lcrash

tests/test_user_blob: tests/test_user_blob.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lcrash

bench/bench_dump: bench/bench_dump.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lcrash

bench/bench_install: bench/bench_install.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lcrash

test: $(TEST_BINS)
	@echo "=== test_basic ==="   && ./tests/test_basic
	@echo "=== test_sigsegv ===" && ./tests/test_sigsegv
	@echo "=== test_sigabrt ===" && ./tests/test_sigabrt
	@echo "=== test_reentry ===" && ./tests/test_reentry
	@echo "=== test_elf ==="     && ./tests/test_elf
	@echo "=== test_fork ==="    && ./tests/test_fork
	@echo "=== test_ymm ==="     && ./tests/test_ymm
	@echo "=== test_waitpid_echild ===" && ./tests/test_waitpid_echild
	@echo "=== test_user_blob ===" && ./tests/test_user_blob

bench: $(BENCH_BINS)
	./bench/bench_dump
	./bench/bench_install

clean:
	rm -f $(LIB) $(LIB_OBJS) $(TEST_BINS) $(TEST_OBJS) $(BENCH_BINS) $(BENCH_OBJS)
