# libpmu: hardware performance counter reads via perf_event_open(2).
# v0.3 adds graceful degradation (dummy ctx) on top of v0.2's rdpmc fast
# path (src/pmu_x86_64.asm) and read(2) path. Public API only adds
# pmu_is_available; pmu_open's failure-mode contract for PMU_ERR_PERM
# changed (non-NULL dummy ctx instead of NULL).

NASM      ?= nasm
CC        ?= gcc
AR        ?= ar
CFLAGS    ?= -std=c11 -Wall -Wextra -Werror -pedantic -O2
NASMFLAGS ?= -f elf64
LDLIBS    ?=

INC := -Iinclude
SRC := src/pmu.c
ASM := src/pmu_x86_64.asm
OBJ := $(SRC:.c=.o) $(ASM:.asm=.o)

LIB := libpmu.a

TESTS := tests/test_basic tests/test_perm tests/test_rdpmc tests/test_edge tests/test_dummy
BENCH := bench/bench_read bench/bench_rdpmc bench/bench_dummy

.PHONY: all test bench clean
all: $(LIB)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

src/%.o: src/%.c include/pmu.h
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

src/%.o: src/%.asm
	$(NASM) $(NASMFLAGS) -o $@ $<

# Tests link the static lib directly so a single `make test` builds everything.
tests/%: tests/%.c $(LIB) include/pmu.h
	$(CC) $(CFLAGS) $(INC) -o $@ $< $(LIB) $(LDLIBS)

bench/%: bench/%.c $(LIB) include/pmu.h
	$(CC) $(CFLAGS) $(INC) -o $@ $< $(LIB) $(LDLIBS) -lrt

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
