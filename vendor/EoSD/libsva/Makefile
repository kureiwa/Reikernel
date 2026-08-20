CC      ?= gcc
AR      ?= ar
CFLAGS  ?= -std=c11 -Wall -Wextra -Werror -pedantic -O2
CPPFLAGS += -D_GNU_SOURCE -Iinclude

SRC_DIR := src
OBJ_DIR := build
TEST_DIR := tests
BENCH_DIR := bench

LIB := libsva.a
OBJS := $(OBJ_DIR)/sva.o

TEST_BINS := \
    $(OBJ_DIR)/test_basic \
    $(OBJ_DIR)/test_guard \
    $(OBJ_DIR)/test_exec \
    $(OBJ_DIR)/test_flush_tlb \
    $(OBJ_DIR)/test_underflow \
    $(OBJ_DIR)/test_hugetlb \
    $(OBJ_DIR)/test_edge

BENCH_BINS := \
    $(OBJ_DIR)/bench_map \
    $(OBJ_DIR)/bench_hugetlb \
    $(OBJ_DIR)/bench_flush_tlb

.PHONY: all test bench clean

all: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Tests
$(OBJ_DIR)/test_%: $(TEST_DIR)/test_%.c $(LIB) | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LIB)

# Bench
$(OBJ_DIR)/bench_%: $(BENCH_DIR)/bench_%.c $(LIB) | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LIB)

test: $(TEST_BINS)
	@set -e; for t in $(TEST_BINS); do \
	    echo "--- $$t ---"; \
	    ./$$t; \
	done

bench: $(BENCH_BINS)
	@set -e; for b in $(BENCH_BINS); do \
	    echo "--- $$b ---"; \
	    ./$$b; \
	done

clean:
	rm -rf $(OBJ_DIR) $(LIB)
