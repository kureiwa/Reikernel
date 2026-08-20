CC ?= gcc
CFLAGS = -std=c11 -Wall -Wextra -Werror -pedantic -O2
AR = ar
ARFLAGS = rcs
INC = -Iinclude

.PHONY: all test bench clean

all: libbarrage.a

libbarrage.a: src/barrage.o
	$(AR) $(ARFLAGS) $@ $^

src/barrage.o: src/barrage.c include/barrage.h
	$(CC) $(CFLAGS) $(INC) -c src/barrage.c -o $@

test: test_basic test_stress test_edge
	./test_basic
	./test_stress
	./test_edge

test_basic: tests/test_basic.c libbarrage.a
	$(CC) $(CFLAGS) $(INC) tests/test_basic.c libbarrage.a -o $@

test_stress: tests/test_stress.c libbarrage.a
	$(CC) $(CFLAGS) $(INC) tests/test_stress.c libbarrage.a -o $@

test_edge: tests/test_edge.c libbarrage.a
	$(CC) $(CFLAGS) $(INC) tests/test_edge.c libbarrage.a -o $@

bench: bench_alloc
	./bench_alloc

bench_alloc: bench/bench_alloc.c libbarrage.a
	$(CC) $(CFLAGS) $(INC) bench/bench_alloc.c libbarrage.a -o $@

clean:
	rm -f libbarrage.a src/barrage.o test_basic test_stress test_edge bench_alloc
