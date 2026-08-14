# libpack: compile-time-described struct serialization with SIMD endianness.
#
# Plain Makefile, no autoconf, no cmake. Builds a static library and the
# associated test and bench binaries. C11 with -Wall -Wextra -Werror
# -pedantic. The baseline CFLAGS carry NO -mavx2 / -mssse3 / -mavx512*
# flags; the SIMD byte-swap variants in pack_simd.c are compiled in via
# __attribute__((target("..."))) and selected at runtime by the ifunc
# resolver in pack_dispatch.c. The resulting library runs on any
# x86-64 baseline CPU and uses the best available SIMD ISA at runtime
# (avx512vbmi > avx2 > ssse3 > scalar).

CC       = gcc
AR       = ar
CFLAGS   = -std=c11 -Wall -Wextra -Werror -pedantic -O2
INCLUDES = -Iinclude

LIB        = libpack.a
LIB_OBJS   = src/pack.o src/pack_simd.o src/pack_dispatch.o

TEST_BINS  = tests/test_basic tests/test_endian tests/test_arrays \
	     tests/test_variable tests/test_allocator tests/test_edge \
	     tests/test_dispatch
TEST_OBJS  = $(TEST_BINS:=.o)
BENCH_BINS = bench/bench_serialize bench/bench_serialize_large bench/bench_variable
BENCH_OBJS = $(BENCH_BINS:=.o)

.PHONY: all test bench clean

all: $(LIB)

# Static library.
$(LIB): $(LIB_OBJS)
	$(AR) rcs $@ $^

# Generic compile rule: any .c -> .o, depends on the public header.
%.o: %.c include/pack.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Test binaries link against the static library.
tests/test_basic:   tests/test_basic.o   $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lpack

tests/test_endian:  tests/test_endian.o  $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lpack

tests/test_arrays:  tests/test_arrays.o  $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lpack

tests/test_variable: tests/test_variable.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lpack

tests/test_allocator: tests/test_allocator.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lpack

tests/test_edge: tests/test_edge.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lpack

tests/test_dispatch: tests/test_dispatch.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lpack

bench/bench_serialize: bench/bench_serialize.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lpack

bench/bench_serialize_large: bench/bench_serialize_large.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lpack

bench/bench_variable: bench/bench_variable.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lpack

# Run all tests; stop on first failure (make stops on nonzero exit).
test: $(TEST_BINS)
	@echo "=== test_basic ==="    && ./tests/test_basic
	@echo "=== test_endian ==="   && ./tests/test_endian
	@echo "=== test_arrays ==="   && ./tests/test_arrays
	@echo "=== test_variable ===" && ./tests/test_variable
	@echo "=== test_allocator ===" && ./tests/test_allocator
	@echo "=== test_edge ==="     && ./tests/test_edge
	@echo "=== test_dispatch ===" && ./tests/test_dispatch

bench: $(BENCH_BINS)
	./bench/bench_serialize
	./bench/bench_serialize_large
	./bench/bench_variable

clean:
	rm -f $(LIB) $(LIB_OBJS) $(TEST_BINS) $(TEST_OBJS) $(BENCH_BINS) $(BENCH_OBJS)
