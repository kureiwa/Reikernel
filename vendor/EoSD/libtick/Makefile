# Makefile for libtick.
#
# Toolchain: gcc (C11), nasm. Static archive libtick.a includes both
# the C object and the NASM object.

NASM      ?= nasm
CC        ?= gcc
AR        ?= ar
CFLAGS    := -std=c11 -Wall -Wextra -Werror -pedantic -O2 -Iinclude
NASMFLAGS := -f elf64
LDLIBS    := -lrt

LIB       := libtick.a
LIB_OBJS  := src/tick.o src/tick_x86_64.o

TEST_BINS  := tests/test_basic tests/test_now tests/test_registry tests/test_stress tests/test_heap tests/test_edge tests/test_drift
BENCH_BINS := bench/bench_sleep bench/bench_registry bench/bench_now

.PHONY: all test bench clean

all: $(LIB)

$(LIB): $(LIB_OBJS)
	$(AR) rcs $@ $^

src/tick.o: src/tick.c include/tick.h
	$(CC) $(CFLAGS) -c $< -o $@

src/tick_x86_64.o: src/tick_x86_64.asm
	$(NASM) $(NASMFLAGS) $< -o $@

tests/%: tests/%.c $(LIB) include/tick.h
	$(CC) $(CFLAGS) $< -L. -ltick $(LDLIBS) -o $@

bench/%: bench/%.c $(LIB) include/tick.h
	$(CC) $(CFLAGS) $< -L. -ltick $(LDLIBS) -o $@

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
