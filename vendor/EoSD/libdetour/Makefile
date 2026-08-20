# libdetour: inline-hook function interception for x86_64 Linux.
#
# Patches the first 14 bytes of a target function with FF 25 00 00 00 00
# + 8-byte absolute address. Displaced prologue instructions are decoded,
# relocated into a trampoline (with RIP-relative fixup), and the
# trampoline ends with a 14-byte absolute jump back into the target past
# the patched bytes. Cross-thread safety via int3-brokered patching.

CC       = gcc
AR       = ar

# Library flags. -O2 per the toolkit convention. -D_GNU_SOURCE for
# dl_iterate_phdr, membarrier, MAP_ANONYMOUS, MAP_32BIT.
CFLAGS_LIB   = -std=c11 -Wall -Wextra -Werror -pedantic -O2 -D_GNU_SOURCE
# Test flags. -O0 so target functions have full classic prologues
# (push rbp / mov rbp,rsp / ...) rather than the 3-byte leaf form that
# -O2 produces. -no-pie so the target lives in low memory where a
# MAP_32BIT trampoline is within +/-2GB (RIP-relative displacement fits
# in 32 bits). -rdynamic exports test symbols to .dynsym so the
# function-boundary check via dl_iterate_phdr finds them.
# -fcf-protection=none so test_basic/test_disable/test_original/test_relocate
# do not start with endbr64 (test_endbr64 overrides this to =full).
# Tests drop -pedantic because the public API takes void* for function
# pointers, which ISO C flags as a function-to-object pointer conversion.
CFLAGS_TEST  = -std=c11 -Wall -Wextra -Werror -O0 -g -D_GNU_SOURCE \
	       -no-pie -rdynamic -fcf-protection=none
# test_edge is built as PIE (-pie -fPIE instead of -no-pie) to exercise
# the MAP_32BIT retry path: the target lives at a high address, so the
# first MAP_32BIT trampoline allocation lands too far from the target's
# RIP-relative data and detour_create must retry with a hint-near
# allocation. -rdynamic exports symbols for the dl_iterate_phdr size
# check; -fcf-protection=none keeps the prologue simple.
CFLAGS_TEST_EDGE = -std=c11 -Wall -Wextra -Werror -O0 -g -D_GNU_SOURCE \
		   -pie -fPIE -rdynamic -fcf-protection=none
# Bench flags. -O2 to measure realistic overhead; -no-pie for RIP-relative
# relocation compatibility with MAP_32BIT trampolines.
CFLAGS_BENCH = -std=c11 -Wall -Wextra -Werror -O2 -D_GNU_SOURCE -no-pie

INC       = -Iinclude
LIB       = libdetour.a
LIB_OBJS  = src/detour.o src/detour_decode.o src/detour_patch.o

TESTS     = tests/test_basic tests/test_original tests/test_disable \
	    tests/test_relocate tests/test_endbr64 tests/test_edge
BENCH     = bench/bench_overhead bench/bench_toggle

LDLIBS    = -ldl

.PHONY: all test bench clean
all: $(LIB)

$(LIB): $(LIB_OBJS)
	$(AR) rcs $@ $^

src/%.o: src/%.c include/detour.h src/detour_internal.h
	$(CC) $(CFLAGS_LIB) $(INC) -c -o $@ $<

tests/test_basic:    tests/test_basic.c    $(LIB) include/detour.h
	$(CC) $(CFLAGS_TEST) $(INC) -o $@ $< $(LIB) $(LDLIBS)

tests/test_original: tests/test_original.c $(LIB) include/detour.h
	$(CC) $(CFLAGS_TEST) $(INC) -o $@ $< $(LIB) $(LDLIBS)

tests/test_disable:  tests/test_disable.c  $(LIB) include/detour.h
	$(CC) $(CFLAGS_TEST) $(INC) -o $@ $< $(LIB) $(LDLIBS)

tests/test_relocate: tests/test_relocate.c $(LIB) include/detour.h
	$(CC) $(CFLAGS_TEST) $(INC) -o $@ $< $(LIB) $(LDLIBS)

# test_endbr64 needs the target compiled with CET enabled so its
# prologue starts with F3 0F 1E FA.
tests/test_endbr64:  tests/test_endbr64.c  $(LIB) include/detour.h
	$(CC) $(CFLAGS_TEST) -fcf-protection=full $(INC) -o $@ $< $(LIB) $(LDLIBS)

# test_edge is built as PIE (see CFLAGS_TEST_EDGE) to exercise the
# MAP_32BIT retry path for high-address targets.
tests/test_edge:     tests/test_edge.c     $(LIB) include/detour.h
	$(CC) $(CFLAGS_TEST_EDGE) $(INC) -o $@ $< $(LIB) $(LDLIBS)

bench/bench_overhead: bench/bench_overhead.c $(LIB) include/detour.h
	$(CC) $(CFLAGS_BENCH) $(INC) -o $@ $< $(LIB) $(LDLIBS)

bench/bench_toggle: bench/bench_toggle.c $(LIB) include/detour.h
	$(CC) $(CFLAGS_BENCH) $(INC) -o $@ $< $(LIB) $(LDLIBS)

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
	rm -f $(LIB) $(LIB_OBJS) $(TESTS) $(BENCH)
