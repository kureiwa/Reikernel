# Makefile for libtopo.
#
# Toolchain: gcc (C11), nasm. Static archive libtopo.a.
#
# libtopo is a zero-dependency leaf module: it links against no other
# EoSD module and no external library beyond libc. The asm helpers in
# src/topo_x86_64.asm are x86_64-only (nasm -f elf64); the C source
# compiles warning-clean under -std=c11 -Wall -Wextra -Werror -pedantic
# and degrades gracefully on non-x86_64 hosts (CPUID-driven paths are
# compiled out, sysfs + getcpu(2) remain).

NASM      ?= nasm
CC        ?= gcc
AR        ?= ar
CFLAGS    ?= -std=c11 -Wall -Wextra -Werror -pedantic -O2
NASMFLAGS ?= -f elf64
LDLIBS    ?=

INC := -Iinclude
SRC := src/topo.c
ASM := src/topo_x86_64.asm
OBJ := $(SRC:.c=.o) $(ASM:.asm=.o)

LIB := libtopo.a

TESTS := tests/test_basic tests/test_affinity tests/test_numa
BENCH := bench/bench_getcpu

.PHONY: all test bench clean
all: $(LIB)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

src/%.o: src/%.c include/topo.h
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

src/%.o: src/%.asm
	$(NASM) $(NASMFLAGS) -o $@ $<

# Tests link the static lib directly so a single `make test` builds
# everything. _GNU_SOURCE is required for sched_getaffinity / CPU_SET
# / syscall(SYS_getcpu).
tests/%: tests/%.c $(LIB) include/topo.h
	$(CC) $(CFLAGS) -D_GNU_SOURCE $(INC) -o $@ $< $(LIB) $(LDLIBS)

bench/%: bench/%.c $(LIB) include/topo.h
	$(CC) $(CFLAGS) -D_GNU_SOURCE $(INC) -o $@ $< $(LIB) $(LDLIBS) -lrt

test: $(TESTS)
	@for t in $(TESTS); do \
		echo "--- $$t ---"; \
		./$$t || exit $$?; \
	done

bench: $(BENCH)
	@for b in $(BENCH); do \
		echo "--- $$b ---"; \
		./$$b || exit $$?; \
	done

clean:
	rm -f $(OBJ) $(LIB) $(TESTS) $(BENCH)
