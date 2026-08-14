# Makefile for libpkey.
#
# Toolchain: gcc (C11), nasm. Static archive libpkey.a.
#
# libpkey wraps Intel Memory Protection Keys (MPK). The asm helpers
# (src/pkey_x86_64.asm: WRPKRU/RDPKRU/CPUID) are x86_64-only; the C
# source is portable but links the asm unconditionally (MPK is x86_64-
# only on Linux). Style mirrors libpmu/libflume: NASM -f elf64, -std=c11
# -Wall -Wextra -Werror -pedantic -O2.

NASM      ?= nasm
CC        ?= gcc
AR        ?= ar
CFLAGS    := -std=c11 -Wall -Wextra -Werror -pedantic -O2 -Iinclude
NASMFLAGS := -f elf64
LDLIBS    :=

LIB       := libpkey.a
LIB_OBJS  := src/pkey.o src/pkey_x86_64.o

TEST_BINS  := tests/test_basic tests/test_protect tests/test_perf
BENCH_BINS := bench/bench_wrpkru bench/bench_mprotect

.PHONY: all test bench clean

all: $(LIB)

$(LIB): $(LIB_OBJS)
	$(AR) rcs $@ $^

src/pkey.o: src/pkey.c include/pkey.h
	$(CC) $(CFLAGS) -c $< -o $@

src/pkey_x86_64.o: src/pkey_x86_64.asm
	$(NASM) $(NASMFLAGS) $< -o $@

# Tests link the static lib directly so a single `make test` builds everything.
tests/%: tests/%.c $(LIB) include/pkey.h
	$(CC) $(CFLAGS) $< -L. -lpkey $(LDLIBS) -o $@

bench/%: bench/%.c $(LIB) include/pkey.h
	$(CC) $(CFLAGS) $< -L. -lpkey $(LDLIBS) -o $@

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do \
		echo "=== $$t ==="; \
		./$$t || exit $$?; \
	done

bench: $(BENCH_BINS)
	@for b in $(BENCH_BINS); do \
		echo "=== $$b ==="; \
		./$$b || exit $$?; \
	done

clean:
	rm -f $(LIB) $(LIB_OBJS) $(TEST_BINS) $(BENCH_BINS)
