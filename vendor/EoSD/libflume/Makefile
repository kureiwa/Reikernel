# Makefile for libflume.
#
# Toolchain: gcc (C11), nasm. Static archive libflume.a.

NASM      ?= nasm
CC        ?= gcc
AR        ?= ar
CFLAGS    := -std=c11 -Wall -Wextra -Werror -pedantic -O2 -pthread -Iinclude
NASMFLAGS := -f elf64
LDFLAGS   := -pthread

LIB       := libflume.a
LIB_OBJS  := src/flume.o src/flume_x86_64.o

TEST_BINS  := tests/test_basic tests/test_mpsc tests/test_batch tests/test_stress
BENCH_BINS := bench/bench_enqueue bench/bench_drain bench/bench_mpsc

.PHONY: all test bench clean

all: $(LIB)

$(LIB): $(LIB_OBJS)
	$(AR) rcs $@ $^

src/flume.o: src/flume.c include/flume.h
	$(CC) $(CFLAGS) -c $< -o $@

src/flume_x86_64.o: src/flume_x86_64.asm
	$(NASM) $(NASMFLAGS) $< -o $@

tests/%: tests/%.c $(LIB) include/flume.h
	$(CC) $(CFLAGS) $< -L. -lflume $(LDFLAGS) -o $@

bench/%: bench/%.c $(LIB) include/flume.h
	$(CC) $(CFLAGS) $< -L. -lflume $(LDFLAGS) -o $@

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
