# EoSD top-level Makefile.

NASM ?= nasm
CC   ?= gcc

# Detect CPU features at build time. Each variable is "yes" or empty.
HAS_AVX    := $(shell grep -q ' avx '    /proc/cpuinfo && echo yes)
HAS_AVX2   := $(shell grep -q ' avx2 '   /proc/cpuinfo && echo yes)
HAS_AVX512 := $(shell grep -q ' avx512f' /proc/cpuinfo && echo yes)
HAS_SSE42  := $(shell grep -q ' sse4_2 ' /proc/cpuinfo && echo yes)
HAS_SSSE3  := $(shell grep -q ' ssse3 '  /proc/cpuinfo && echo yes)

# Export so module Makefiles can use them.
export HAS_AVX HAS_AVX2 HAS_AVX512 HAS_SSE42 HAS_SSSE3

MODULES = libbarrage libcrash libdetour libflume libpack libpmu libpkey libspinit libspoon libsva libtick libtopo liburing

.PHONY: all test bench extreme tools clean $(MODULES)

all: $(MODULES)

$(MODULES):
	$(MAKE) -C $@

test:
	@failed=0; \
	for m in $(MODULES); do \
		echo "======== $$m ========"; \
		$(MAKE) -C $$m test || failed=$$((failed + 1)); \
	done; \
	if [ $$failed -gt 0 ]; then \
		echo ""; \
		echo "FAIL: $$failed module(s) failed tests"; \
		exit 1; \
	fi; \
	echo ""; \
	echo "All $(words $(MODULES)) modules passed."

bench:
	@failed=0; \
	for m in $(MODULES); do \
		echo "======== $$m ========"; \
		$(MAKE) -C $$m bench || failed=$$((failed + 1)); \
	done; \
	if [ $$failed -gt 0 ]; then \
		echo ""; \
		echo "FAIL: $$failed module(s) failed bench"; \
		exit 1; \
	fi

extreme: all
	@$(MAKE) -C tests/extreme test

tools: all
	@$(MAKE) -C tools

clean:
	@for m in $(MODULES); do \
		$(MAKE) -C $$m clean 2>/dev/null; \
	done
	$(MAKE) -C tools clean 2>/dev/null
	rm -rf example/build tests/extreme/test_*_extreme
