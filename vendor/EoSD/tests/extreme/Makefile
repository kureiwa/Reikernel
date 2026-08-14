# tests/extreme/Makefile — builds and runs extreme stress tests.
#
# Each test links against the module static libraries (assumed already
# built via `make` at the repo root).

CC       = gcc
CFLAGS   = -std=c11 -Wall -Wextra -Werror -pedantic -O2 -D_GNU_SOURCE
CPPFLAGS = -I../../libbarrage/include -I../../libcrash/include \
	   -I../../libdetour/include -I../../libpack/include \
	   -I../../libpmu/include -I../../libspinit/include \
	   -I../../libspoon/include -I../../libsva/include \
	   -I../../libtick/include

LDFLAGS  = -L../../libbarrage -L../../libcrash -L../../libdetour \
	   -L../../libpack -L../../libpmu -L../../libspinit \
	   -L../../libspoon -L../../libsva -L../../libtick
LDLIBS   = -lrt -ldl -lpthread

TESTS = test_barrage_extreme test_crash_extreme test_detour_extreme \
	test_pack_extreme test_pmu_extreme test_spinit_extreme \
	test_spoon_extreme test_sva_extreme test_tick_extreme

.PHONY: all test clean

all: $(TESTS)

# All tests depend on the shared latency header.
$(TESTS): latency.h

test_barrage_extreme: test_barrage_extreme.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -lbarrage $(LDLIBS)

test_crash_extreme: test_crash_extreme.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -lcrash $(LDLIBS)

test_detour_extreme: test_detour_extreme.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -O0 -g -no-pie -rdynamic -fcf-protection=none \
	    -o $@ $< $(LDFLAGS) -ldetour $(LDLIBS)

test_pack_extreme: test_pack_extreme.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -lpack $(LDLIBS)

test_pmu_extreme: test_pmu_extreme.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -lpmu $(LDLIBS)

test_spinit_extreme: test_spinit_extreme.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -lspinit $(LDLIBS)

test_spoon_extreme: test_spoon_extreme.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -lspoon $(LDLIBS)

test_sva_extreme: test_sva_extreme.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -lsva $(LDLIBS)

test_tick_extreme: test_tick_extreme.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -ltick $(LDLIBS)

test: all
	@failed=0; \
	for t in $(TESTS); do \
	        echo "======== $$t ========"; \
	        ./$$t || failed=$$((failed + 1)); \
	        echo ""; \
	done; \
	if [ $$failed -gt 0 ]; then \
	        echo "FAIL: $$failed extreme test(s) failed"; \
	        exit 1; \
	fi; \
	echo "All extreme tests passed."

clean:
	rm -f $(TESTS)
