# Makefile for libspinit.
#
# Toolchain: gcc (C11), nasm. Static archive libspinit.a.

NASM      ?= nasm
CC        ?= gcc
AR        ?= ar
CFLAGS    := -std=c11 -Wall -Wextra -Werror -pedantic -O2 -D_DEFAULT_SOURCE -pthread -Iinclude
NASMFLAGS := -f elf64
LDFLAGS   := -pthread

LIB       := libspinit.a
LIB_OBJS  := src/spinit.o src/spinit_x86_64.o

TEST_BINS  := tests/test_basic tests/test_contended tests/test_stress tests/test_edge tests/test_backoff
BENCH_BINS := bench/bench_uncontended bench/bench_contended bench/bench_backoff

.PHONY: all test bench clean

all: $(LIB)

$(LIB): $(LIB_OBJS)
	$(AR) rcs $@ $^

src/spinit.o: src/spinit.c include/spinit.h
	$(CC) $(CFLAGS) -c $< -o $@

src/spinit_x86_64.o: src/spinit_x86_64.asm
	$(NASM) $(NASMFLAGS) $< -o $@

tests/%: tests/%.c $(LIB) include/spinit.h
	$(CC) $(CFLAGS) $< -L. -lspinit $(LDFLAGS) -o $@

bench/%: bench/%.c $(LIB) include/spinit.h
	$(CC) $(CFLAGS) $< -L. -lspinit $(LDFLAGS) -o $@

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do \
	        echo "=== $$t ==="; \
	        ./$$t || exit 1; \
	done

bench: $(BENCH_BINS)
	@for b in $(BENCH_BINS); do \
	        echo "=== $$b ==="; \
	        ./$$b || exit 1; \
	done

clean:
	rm -f $(LIB) $(LIB_OBJS) $(TEST_BINS) $(BENCH_BINS)
