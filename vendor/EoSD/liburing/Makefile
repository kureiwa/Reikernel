# Makefile for liburing.
#
# Toolchain: gcc (C11), nasm. Static archive liburing.a.
#
# liburing does not link against any other EoSD module. The static
# archive contains uring.o + uring_x86_64.o; tests and benches link
# liburing.a directly so a single `make test` builds everything.

NASM      ?= nasm
CC        ?= gcc
AR        ?= ar
CFLAGS    := -std=c11 -Wall -Wextra -Werror -pedantic -O2 -pthread -Iinclude
NASMFLAGS := -f elf64
LDFLAGS   := -pthread
LDLIBS    :=

LIB       := liburing.a
LIB_OBJS  := src/uring.o src/uring_x86_64.o

TEST_BINS  := tests/test_basic tests/test_rw tests/test_batch
BENCH_BINS := bench/bench_submit bench/bench_roundtrip

.PHONY: all test bench clean

all: $(LIB)

$(LIB): $(LIB_OBJS)
	$(AR) rcs $@ $^

src/uring.o: src/uring.c include/uring.h
	$(CC) $(CFLAGS) -c $< -o $@

src/uring_x86_64.o: src/uring_x86_64.asm
	$(NASM) $(NASMFLAGS) $< -o $@

tests/%: tests/%.c $(LIB) include/uring.h
	$(CC) $(CFLAGS) $< -L. -luring $(LDFLAGS) -o $@

bench/%: bench/%.c $(LIB) include/uring.h
	$(CC) $(CFLAGS) $< -L. -luring $(LDFLAGS) -o $@

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
