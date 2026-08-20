# example/Makefile — builds cross-module integration examples.
#
# Each example links against the static libraries of the modules it uses.
# Assumes each module's .a has already been built (run `make` at the
# repo root first).

CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -Werror -pedantic -O2 -D_GNU_SOURCE
CPPFLAGS = -I../libbarrage/include -I../libcrash/include -I../libdetour/include \
           -I../libpack/include -I../libpmu/include -I../libspinit/include \
           -I../libspoon/include -I../libsva/include -I../libtick/include

LDFLAGS = -L../libbarrage -L../libcrash -L../libdetour -L../libpack \
          -L../libpmu -L../libspinit -L../libspoon -L../libsva -L../libtick
LDLIBS  = -lrt -ldl -lpthread

EXAMPLES = spoon_sva tick_spoon barrage_pack spinit_pmu crash_spoon_sva

.PHONY: all clean run

all: $(EXAMPLES)

spoon_sva: spoon_sva.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -lspoon -lsva $(LDLIBS)

tick_spoon: tick_spoon.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -ltick -lspoon $(LDLIBS)

barrage_pack: barrage_pack.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -lbarrage -lpack $(LDLIBS)

spinit_pmu: spinit_pmu.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -lspinit -lpmu $(LDLIBS)

crash_spoon_sva: crash_spoon_sva.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -lcrash -lspoon -lsva $(LDLIBS)

run: all
	@for ex in $(EXAMPLES); do \
		echo "======== $$ex ========"; \
		./$$ex || echo "  (exited $$?)"; \
		echo ""; \
	done

clean:
	rm -f $(EXAMPLES)
