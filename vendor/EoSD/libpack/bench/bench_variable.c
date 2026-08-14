/* Enable POSIX clock_gettime / CLOCK_MONOTONIC. The library itself
 * (pack.c, pack_simd.c) stays pure C11; only the bench needs POSIX. */
#define _POSIX_C_SOURCE 199309L

#include <pack.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Throughput benchmark for the v0.2 variable-length paths.
 *
 * The struct carries a 4-KiB PACK_BYTES payload plus a 256-element
 * PACK_VAR_ARRAY of uint32_t. The serialize path is dominated by:
 *   - one 4-byte length prefix + 4 KiB memcpy for the bytes field
 *   - one 4-byte length prefix + 1 KiB SIMD byte-swap for the var array
 *
 * 200,000 round-trip iterations. Reports ns/op and GB/s (decimal,
 * 1e9 bytes/sec) for serialize and deserialize separately, for both
 * the caller-supplied-buffer entry point and the allocator entry point.
 * The __asm__ volatile memory clobber after each call prevents the
 * compiler from hoisting the call out of the loop.
 */

typedef struct {
    uint8_t  *payload;
    size_t    payload_len;
    uint32_t *vals;
    size_t    vals_len;
    uint32_t  tag;
} var_bench_t;

static const pack_field_t vb_fields[] = {
    PACK_VAR_FIELD(var_bench_t, payload, PACK_BYTES,     payload_len),
    PACK_VAR_FIELD(var_bench_t, vals,    PACK_VAR_ARRAY, vals_len),
    PACK_FIELD    (var_bench_t, tag,     PACK_U32),
};

static const pack_schema_t vb_schema = {
    .fields = vb_fields,
    .field_count = sizeof(vb_fields) / sizeof(vb_fields[0]),
    .struct_size = sizeof(var_bench_t),
};

#define ITERS 200000u
#define PAYLOAD_BYTES 4096u
#define VALS_COUNT    256u

static double elapsed_secs(const struct timespec *t0, const struct timespec *t1)
{
    return (double)(t1->tv_sec - t0->tv_sec) +
           (double)(t1->tv_nsec - t0->tv_nsec) * 1e-9;
}

int main(void)
{
    var_bench_t in;
    static uint8_t payload_buf[PAYLOAD_BYTES];
    static uint32_t vals_buf[VALS_COUNT];
    for (uint32_t i = 0; i < PAYLOAD_BYTES; i++) {
        payload_buf[i] = (uint8_t)i;
    }
    for (uint32_t i = 0; i < VALS_COUNT; i++) {
        vals_buf[i] = i * 0x01010101u;
    }
    in.payload = payload_buf;
    in.payload_len = PAYLOAD_BYTES;
    in.vals = vals_buf;
    in.vals_len = VALS_COUNT;
    in.tag = 0xBEEFCAFEu;

    const size_t wire_size = pack_serialized_size_var(&vb_schema, &in);
    printf("libpack v0.2 variable-length benchmark\n");
    printf("  struct size:        %zu bytes\n", sizeof(var_bench_t));
    printf("  wire size:          %zu bytes (4+%u + 4+%u + 4)\n",
           wire_size, PAYLOAD_BYTES, VALS_COUNT * 4u);
    printf("  iterations:         %u\n", ITERS);
    printf("  fields:             payload (PACK_BYTES %u B), vals (PACK_VAR_ARRAY %u x uint32), tag (PACK_U32)\n",
           PAYLOAD_BYTES, VALS_COUNT);

    /* Caller-supplied-buffer path. */
    unsigned char *buf = malloc(wire_size);
    if (!buf) {
        printf("  ERROR: malloc failed\n");
        return 1;
    }
    static uint8_t out_payload[PAYLOAD_BYTES];
    static uint32_t out_vals[VALS_COUNT];
    var_bench_t out;
    out.payload = out_payload;
    out.vals = out_vals;

    struct timespec t0, t1;
    size_t written = 0;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t i = 0; i < ITERS; i++) {
        pack_serialize(&vb_schema, &in, buf, wire_size, &written);
        __asm__ volatile("" : : : "memory");
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ser_secs = elapsed_secs(&t0, &t1);
    double ser_ns = ser_secs * 1e9 / ITERS;
    double ser_gbs = (double)written * ITERS / ser_secs / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t i = 0; i < ITERS; i++) {
        pack_deserialize(&vb_schema, &out, buf, written);
        __asm__ volatile("" : : : "memory");
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double de_secs = elapsed_secs(&t0, &t1);
    double de_ns = de_secs * 1e9 / ITERS;
    double de_gbs = (double)written * ITERS / de_secs / 1e9;

    printf("\n  caller-supplied buffer:\n");
    printf("    serialize:   %7.2f ns/op   %6.2f GB/s\n", ser_ns, ser_gbs);
    printf("    deserialize: %7.2f ns/op   %6.2f GB/s\n", de_ns, de_gbs);

    /* Allocator path. */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t i = 0; i < ITERS; i++) {
        void *sb = NULL;
        size_t ss = 0;
        pack_serialize_alloc(&vb_schema, &in, &sb, &ss, NULL);
        __asm__ volatile("" : : : "memory");
        free(sb);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ser_a_secs = elapsed_secs(&t0, &t1);
    double ser_a_ns = ser_a_secs * 1e9 / ITERS;
    double ser_a_gbs = (double)written * ITERS / ser_a_secs / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t i = 0; i < ITERS; i++) {
        void *os = NULL;
        pack_deserialize_alloc(&vb_schema, buf, written, &os, NULL);
        __asm__ volatile("" : : : "memory");
        pack_free_struct(&vb_schema, os, NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double de_a_secs = elapsed_secs(&t0, &t1);
    double de_a_ns = de_a_secs * 1e9 / ITERS;
    double de_a_gbs = (double)written * ITERS / de_a_secs / 1e9;

    printf("\n  allocator path (malloc/free):\n");
    printf("    serialize:   %7.2f ns/op   %6.2f GB/s\n", ser_a_ns, ser_a_gbs);
    printf("    deserialize: %7.2f ns/op   %6.2f GB/s\n", de_a_ns, de_a_gbs);

    /* Sanity check. */
    if (out.tag != in.tag ||
        memcmp(out.payload, in.payload, PAYLOAD_BYTES) != 0 ||
        memcmp(out.vals, in.vals, VALS_COUNT * sizeof(uint32_t)) != 0) {
        printf("  ERROR: round-trip mismatch\n");
        free(buf);
        return 1;
    }

    free(buf);
    return 0;
}
