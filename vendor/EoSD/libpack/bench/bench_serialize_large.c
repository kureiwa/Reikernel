/* Enable POSIX clock_gettime / CLOCK_MONOTONIC. The library itself
 * (pack.c, pack_simd.c) stays pure C11; only the bench needs POSIX. */
#define _POSIX_C_SOURCE 199309L

#include <pack.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

/*
 * Larger-struct throughput benchmark for pack_serialize /
 * pack_deserialize, added to amortize the per-call schema-walk and
 * ifunc-dispatch overhead that dominates bench_serialize's 64-byte
 * measurement.
 *
 * Three struct sizes are exercised, each as a single PACK_ARRAY of
 * uint32_t:
 *
 *   -  256 bytes (64 uint32_t): 1 VPERMB zmm + 1 VPERMB zmm + scalar
 *                               tail (avx512), or 2 VPSHUFB ymm + tail
 *                               (avx2). The per-call overhead is now
 *                               amortized over 4x the work of the
 *                               64-byte bench.
 *   - 1024 bytes (256 uint32_t): 16 VPERMB zmm (avx512) or 32 VPSHUFB
 *                                ymm (avx2). SIMD throughput starts
 *                                to dominate.
 *   - 4096 bytes (1024 uint32_t): 64 VPERMB zmm or 128 VPSHUFB ymm.
 *                                 Per-call overhead is negligible;
 *                                 this is the asymptotic SIMD
 *                                 throughput.
 *
 * 200,000 round-trip iterations per size. Reports ns/op and GB/s
 * (decimal, 1e9 bytes/sec) for each direction separately.
 *
 * The __asm__ volatile memory clobber after each call prevents the
 * compiler from hoisting the call out of the loop.
 */

#define ITERS 200000u

typedef struct { uint32_t vals[64];    } bench_256_t;
typedef struct { uint32_t vals[256];   } bench_1024_t;
typedef struct { uint32_t vals[1024];  } bench_4096_t;

static const pack_field_t f256[]  = { PACK_ARRAY_FIELD(bench_256_t,  vals) };
static const pack_field_t f1024[] = { PACK_ARRAY_FIELD(bench_1024_t, vals) };
static const pack_field_t f4096[] = { PACK_ARRAY_FIELD(bench_4096_t, vals) };

static const pack_schema_t schema_256  = {
    .fields = f256,  .field_count = 1, .struct_size = sizeof(bench_256_t),
};
static const pack_schema_t schema_1024 = {
    .fields = f1024, .field_count = 1, .struct_size = sizeof(bench_1024_t),
};
static const pack_schema_t schema_4096 = {
    .fields = f4096, .field_count = 1, .struct_size = sizeof(bench_4096_t),
};

static double elapsed_secs(const struct timespec *t0, const struct timespec *t1)
{
    return (double)(t1->tv_sec - t0->tv_sec) +
           (double)(t1->tv_nsec - t0->tv_nsec) * 1e-9;
}

/* Returns 0 on success, 1 on round-trip mismatch. */
static int bench_one(const pack_schema_t *schema, size_t struct_size,
                     const char *label)
{
    static uint8_t inbuf[4096];
    static uint8_t outbuf[4096];
    static uint8_t wire[4096];

    uint32_t *in = (uint32_t *)(void *)inbuf;
    for (size_t i = 0; i < struct_size / 4; i++) {
        in[i] = (uint32_t)i * 0x11111111u;
    }

    size_t written = 0;
    struct timespec t0, t1;

    /* Serialize. */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t i = 0; i < ITERS; i++) {
        pack_serialize(schema, inbuf, wire, sizeof(wire), &written);
        __asm__ volatile("" : : : "memory");
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ser_secs = elapsed_secs(&t0, &t1);
    double ser_ns = ser_secs * 1e9 / ITERS;
    double ser_gbs = (double)written * ITERS / ser_secs / 1e9;

    /* Deserialize. */
    memset(outbuf, 0, sizeof(outbuf));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t i = 0; i < ITERS; i++) {
        pack_deserialize(schema, outbuf, wire, written);
        __asm__ volatile("" : : : "memory");
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double de_secs = elapsed_secs(&t0, &t1);
    double de_ns = de_secs * 1e9 / ITERS;
    double de_gbs = (double)written * ITERS / de_secs / 1e9;

    printf("  %-4s (%4zu B wire):  serialize %7.2f ns/op  %6.2f GB/s   |   "
           "deserialize %7.2f ns/op  %6.2f GB/s\n",
           label, written, ser_ns, ser_gbs, de_ns, de_gbs);

    /* Sanity check. */
    if (memcmp(inbuf, outbuf, struct_size) != 0) {
        printf("  ERROR: %s round-trip mismatch\n", label);
        return 1;
    }
    return 0;
}

int main(void)
{
    printf("libpack large-struct serialize/deserialize benchmark\n");
    printf("  iterations per size: %u\n", ITERS);
    printf("  schema: one PACK_ARRAY of uint32_t (no version prefix)\n");
    printf("\n");

    int rc = 0;
    rc |= bench_one(&schema_256,  sizeof(bench_256_t),  "256B");
    rc |= bench_one(&schema_1024, sizeof(bench_1024_t), "1KB");
    rc |= bench_one(&schema_4096, sizeof(bench_4096_t), "4KB");

    if (rc != 0) {
        return 1;
    }
    return 0;
}
