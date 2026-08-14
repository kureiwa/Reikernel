# tools/Makefile: build the standalone libcrash dump decoder.
#
# Plain Makefile, no autoconf, no cmake. C11 with -Wall -Wextra -Werror
# -pedantic. The crashdump binary is fully standalone: it does NOT link
# against libcrash, it parses the dump format directly by offset.
#
# `make test` invokes test_crashdump.sh, which builds libcrash, runs a
# test that produces a dump file in /tmp, then runs crashdump on it and
# greps the output for the expected fields.

CC      ?= gcc
CFLAGS  ?= -std=c11 -Wall -Wextra -Werror -pedantic -O2
LDFLAGS ?=

BIN   = crashdump
SRC   = crashdump.c

LIBCRASH_DIR = ../libcrash

.PHONY: all test clean

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test: $(BIN)
	@./test_crashdump.sh "$(LIBCRASH_DIR)" ./$(BIN)

clean:
	rm -f $(BIN)
