/* Enable POSIX clock_gettime / CLOCK_MONOTONIC. The library itself
 * (pack.c, pack_simd.c) stays pure C11; only the bench needs POSIX. */
#define _POSIX_C_SOURCE 199309L

#include <pack.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

/*
 * Throughput benchmark for pack_serialize and pack_deserialize.
 *
 * The struct is 64 bytes (16 uint32_t elements in one array field). On
 * a host with AVX-512 VBMI the resolver selects pack_bswap_uint32_array
 * _avx512, so the SIMD work per call is exactly one VPERMB zmm (16
 * uint32_t = 64 bytes). On AVX2-only hosts it is two VPSHUFB ymm; on
 * SSSE3-only hosts, four PSHUFB xmm. The schema-walk + memcpy + ifunc
 * dispatch dominate the wall-clock time at this size, which is the
 * point: this bench measures per-call overhead, not SIMD throughput.
 *
 * See bench_serialize_large.c for the same code path on 256-byte,
 * 1 KB, and 4 KB structs, where the SIMD work dominates and the
 * reported GB/s approaches the asymptotic byte-swap throughput.
 *
 * 1,000,000 round-trip iterations. Reports ns/op and GB/s (decimal,
 * 1e9 bytes/sec) for each direction separately. The __asm__ volatile
 * memory clobber after each call prevents the compiler from hoisting
 * the call out of the loop (the input is loop-invariant, so without
 * the barrier the optimizer could lift pack_serialize above the loop).
 */

typedef struct {
    uint32_t vals[16];
} bench_t;

static const pack_field_t bench_fields[] = {
    PACK_ARRAY_FIELD(bench_t, vals),
};

static const pack_schema_t bench_schema = {
    .fields = bench_fields,
    .field_count = sizeof(bench_fields) / sizeof(bench_fields[0]),
    .struct_size = sizeof(bench_t),
};

#define ITERS 1000000u

static double elapsed_secs(const struct timespec *t0, const struct timespec *t1)
{
    return (double)(t1->tv_sec - t0->tv_sec) +
           (double)(t1->tv_nsec - t0->tv_nsec) * 1e-9;
}

int main(void)
{
    bench_t in;
    for (int i = 0; i < 16; i++) {
        in.vals[i] = (uint32_t)i * 0x11111111u;
    }

    unsigned char buf[64];
    size_t written = 0;

    struct timespec t0, t1;

    /* Serialize 1M times. */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t i = 0; i < ITERS; i++) {
        pack_serialize(&bench_schema, &in, buf, sizeof(buf), &written);
        __asm__ volatile("" : : : "memory");
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ser_secs = elapsed_secs(&t0, &t1);
    double ser_ns_per_op = ser_secs * 1e9 / ITERS;
    double ser_gb_per_sec =
        (double)written * (double)ITERS / ser_secs / 1e9;

    /* Deserialize 1M times. */
    bench_t out;
    memset(&out, 0, sizeof(out));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t i = 0; i < ITERS; i++) {
        pack_deserialize(&bench_schema, &out, buf, written);
        __asm__ volatile("" : : : "memory");
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double de_secs = elapsed_secs(&t0, &t1);
    double de_ns_per_op = de_secs * 1e9 / ITERS;
    double de_gb_per_sec =
        (double)written * (double)ITERS / de_secs / 1e9;

    printf("libpack serialize/deserialize benchmark\n");
    printf("  struct size:        %zu bytes\n", sizeof(bench_t));
    printf("  serialized size:    %zu bytes\n", pack_serialized_size(&bench_schema));
    printf("  iterations:         %u\n", ITERS);
    printf("  field count:        %zu (one array of 16 uint32_t)\n",
           bench_schema.field_count);
    printf("\n");
    printf("  serialize:   %7.2f ns/op   %6.2f GB/s\n",
           ser_ns_per_op, ser_gb_per_sec);
    printf("  deserialize: %7.2f ns/op   %6.2f GB/s\n",
           de_ns_per_op, de_gb_per_sec);

    /* Sanity-check the last deserialize result; also prevents the
     * compiler from deleting the deserialize loop as dead code. */
    if (out.vals[0] != in.vals[0] || out.vals[15] != in.vals[15]) {
        printf("  ERROR: round-trip mismatch\n");
        return 1;
    }

    (void)written;
    return 0;
}
