NASM      ?= nasm
CC        ?= gcc
CFLAGS    = -std=c11 -Wall -Wextra -Werror -pedantic -O2 -D_GNU_SOURCE
CPPFLAGS  = -Iinclude -Isrc
AR        = ar
ARFLAGS   = rcs

LIB       = libspoon.a
LIB_OBJS  = src/spoon.o src/spoon_x86_64.o

TESTS     = tests/test_basic tests/test_yield tests/test_fp tests/test_stress tests/test_edge
TEST_BINS = $(TESTS:=.bin)
BENCH     = bench/bench_switch

.PHONY: all test bench clean

all: $(LIB)

$(LIB): $(LIB_OBJS)
	$(AR) $(ARFLAGS) $@ $^

src/spoon.o: src/spoon.c src/spoon_internal.h include/spoon.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

src/spoon_x86_64.o: src/spoon_x86_64.asm
	$(NASM) -f elf64 -o $@ $<

tests/%.o: tests/%.c include/spoon.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

tests/%.bin: tests/%.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lspoon

bench/%.o: bench/%.c include/spoon.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

$(BENCH): bench/bench_switch.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lspoon

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do \
	        ./$$t || exit 1; \
	done
	@echo "All spoon tests passed."

bench: $(BENCH)
	./$(BENCH)

clean:
	rm -f $(LIB) $(LIB_OBJS) $(TESTS:=.o) $(TEST_BINS) bench/*.o $(BENCH)
