# Reikernel top-level Makefile.
#
# Builds vendor/EoSD first (so libbarrage.a + libtopo.a exist), then compiles
# c/*.c against them into libreikernel.so.
#
# Targets:
#   make             -- build libreikernel.so (picks up all c/*.c automatically,
#                       including rk_norm.c, rk_matmul.c, rk_softmax.c,
#                       rk_layernorm.c, rk_turbo.c)
#   make vendor     -- build just vendor/EoSD
#   make test       -- run the Python correctness tests for v0.1 (rms_norm)
#   make test-mm    -- run the Python correctness tests for v0.2 (mm)
#   make test-softmax -- run the Python correctness tests for v0.3 (softmax)
#   make test-layernorm -- run the Python correctness tests for v0.4 (layer_norm)
#   make test-turbo -- run the Python tests for v0.5 (turbo context manager)
#   make bench      -- run the v0.1 microbenchmark (rms_norm)
#   make bench-mm   -- run the v0.2 microbenchmark (mm)
#   make bench-softmax -- run the v0.3 microbenchmark (softmax)
#   make bench-layernorm -- run the v0.4 microbenchmark (layer_norm)
#   make bench-turbo -- run the v0.5 smoke benchmark (turbo on vs off)
#   make clean      -- remove build artifacts (keeps vendored source)
#
# Requirements: gcc (C11), nasm 2.16+ (for EoSD's asm), Python 3.10+, PyTorch.
# nasm lives at /home/z/.local/bin/nasm; we put that on PATH for the build.

NASM_PATH := /home/z/.local/bin
PATH      := $(NASM_PATH):$(PATH)

CC      ?= cc
CFLAGS  ?= -std=c11 -Wall -Wextra -O3 -march=native -fopenmp -fPIC
LDFLAGS ?= -shared -fopenmp

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
VENDOR_DIR    := $(REPO_ROOT)/vendor/EoSD
BUILD_DIR     := $(REPO_ROOT)/build
C_SRC_DIR     := $(REPO_ROOT)/c
PYTHON_DIR    := $(REPO_ROOT)/python
TESTS_DIR     := $(REPO_ROOT)/tests
BENCH_DIR     := $(REPO_ROOT)/bench

LIB_NAME      := libreikernel.so
LIB_PATH      := $(BUILD_DIR)/$(LIB_NAME)

BARRAGE_LIB   := $(VENDOR_DIR)/libbarrage/libbarrage.a
BARRAGE_INC   := $(VENDOR_DIR)/libbarrage/include
TOPO_LIB      := $(VENDOR_DIR)/libtopo/libtopo.a
TOPO_INC      := $(VENDOR_DIR)/libtopo/include

C_SOURCES     := $(wildcard $(C_SRC_DIR)/*.c)
C_OBJECTS     := $(patsubst $(C_SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SOURCES))

.PHONY: all vendor test test-mm test-softmax test-layernorm test-turbo bench bench-mm bench-softmax bench-layernorm bench-turbo clean

all: $(LIB_PATH)

# The vendor build is keyed off BARRAGE_LIB (the first EoSD module we
# consume). The EoSD top-level Makefile builds all 13 modules in one pass,
# so by the time libbarrage.a exists, libtopo.a (and every other EoSD
# static lib we link against in v0.5+) also exists.
#
# Sanity check: tell the user where nasm is if it's missing.
$(BARRAGE_LIB):
	@command -v nasm >/dev/null 2>&1 || { \
	        echo "ERROR: nasm not found on PATH."; \
	        echo "  nasm is expected at $(NASM_PATH)/nasm (built from source)."; \
	        echo "  Put $(NASM_PATH) on PATH before running make, or set NASM_PATH."; \
	        exit 1; }
	@PATH=$(NASM_PATH):$$PATH $(MAKE) -C $(VENDOR_DIR)

vendor: $(BARRAGE_LIB)

$(BUILD_DIR):
	@mkdir -p $@

# rk_turbo.c (and any future libtopo consumer) needs topo.h on the
# include path. Other c/*.c files just don't include it; the extra -I
# is harmless.
$(BUILD_DIR)/%.o: $(C_SRC_DIR)/%.c | $(BUILD_DIR) $(BARRAGE_LIB) $(TOPO_LIB)
	@echo "  CC  $<"
	@$(CC) $(CFLAGS) -I$(C_SRC_DIR) -I$(BARRAGE_INC) -I$(TOPO_INC) -c $< -o $@

# Link against both libbarrage.a (per-thread arenas for rms_norm / mm /
# softmax / layer_norm) and libtopo.a (CPU topology + pinning for
# rk_turbo_enter / rk_turbo_exit). libtopo is zero-dep (libc only),
# so no extra -l flags are needed; libbarrage is the same.
$(LIB_PATH): $(C_OBJECTS) $(BARRAGE_LIB) $(TOPO_LIB) | $(BUILD_DIR)
	@echo "  LD  $@"
	@$(CC) $(LDFLAGS) -o $@ $(C_OBJECTS) $(BARRAGE_LIB) $(TOPO_LIB) -lm
	@echo "Built $(LIB_PATH)"

test: $(LIB_PATH)
	@echo "=== Running correctness tests (v0.1: rms_norm) ==="
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
	 PYTHONPATH=$(PYTHON_DIR):$(REPO_ROOT):$$PYTHONPATH \
	 python3 $(TESTS_DIR)/test_rms_norm.py

test-mm: $(LIB_PATH)
	@echo "=== Running correctness tests (v0.2: mm) ==="
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
	 PYTHONPATH=$(PYTHON_DIR):$(REPO_ROOT):$$PYTHONPATH \
	 python3 $(TESTS_DIR)/test_mm.py

bench: $(LIB_PATH)
	@echo "=== Running microbenchmark (v0.1: rms_norm) ==="
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
	 PYTHONPATH=$(PYTHON_DIR):$(REPO_ROOT):$$PYTHONPATH \
	 python3 $(BENCH_DIR)/bench_rms_norm.py

bench-mm: $(LIB_PATH)
	@echo "=== Running microbenchmark (v0.2: mm) ==="
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
	 PYTHONPATH=$(PYTHON_DIR):$(REPO_ROOT):$$PYTHONPATH \
	 python3 $(BENCH_DIR)/bench_mm.py

test-softmax: $(LIB_PATH)
	@echo "=== Running correctness tests (v0.3: softmax) ==="
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
	 PYTHONPATH=$(PYTHON_DIR):$(REPO_ROOT):$$PYTHONPATH \
	 python3 $(TESTS_DIR)/test_softmax.py

bench-softmax: $(LIB_PATH)
	@echo "=== Running microbenchmark (v0.3: softmax) ==="
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
	 PYTHONPATH=$(PYTHON_DIR):$(REPO_ROOT):$$PYTHONPATH \
	 python3 $(BENCH_DIR)/bench_softmax.py

test-layernorm: $(LIB_PATH)
	@echo "=== Running correctness tests (v0.4: layer_norm) ==="
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
	 PYTHONPATH=$(PYTHON_DIR):$(REPO_ROOT):$$PYTHONPATH \
	 python3 $(TESTS_DIR)/test_layernorm.py

bench-layernorm: $(LIB_PATH)
	@echo "=== Running microbenchmark (v0.4: layer_norm) ==="
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
	 PYTHONPATH=$(PYTHON_DIR):$(REPO_ROOT):$$PYTHONPATH \
	 python3 $(BENCH_DIR)/bench_layernorm.py

test-turbo: $(LIB_PATH)
	@echo "=== Running tests (v0.5: turbo context manager) ==="
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
	 PYTHONPATH=$(PYTHON_DIR):$(REPO_ROOT):$$PYTHONPATH \
	 python3 $(TESTS_DIR)/test_turbo.py

bench-turbo: $(LIB_PATH)
	@echo "=== Running smoke benchmark (v0.5: turbo on vs off) ==="
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
	 PYTHONPATH=$(PYTHON_DIR):$(REPO_ROOT):$$PYTHONPATH \
	 python3 $(BENCH_DIR)/bench_turbo.py

clean:
	@echo "Cleaning Reikernel build artifacts..."
	@rm -rf $(BUILD_DIR)
	@PATH=$(NASM_PATH):$$PATH $(MAKE) -C $(VENDOR_DIR) clean >/dev/null 2>&1 || true
	@echo "Cleaned."
