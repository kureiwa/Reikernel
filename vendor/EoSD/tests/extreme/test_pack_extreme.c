/* libpack extreme tests: push fixed-layout serialization to its limits.
 *
 * Tests:
 * - Struct with every scalar + array field type (13 fields: 10 scalars + 3 arrays)
 * - Fuzz round-trip (10K random fills of that fixed-layout struct; does
 *   not exercise variable-length schemas -- see tests/test_edge.c for that)
 * - Array edge cases (0-length, exact SIMD width, SIMD+1)
 * - All field types in one struct
 * - Truncated input (deserialize with short buffer)
 */

#include <pack.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "latency.h"

/* Struct with every field type. */
typedef struct {
    uint8_t  u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    int8_t   i8;
    int16_t  i16;
    int32_t  i32;
    int64_t  i64;
    float    f32;
    double   f64;
    uint32_t arr8[8];    /* exact SIMD width for uint32 */
    uint64_t arr4[4];    /* exact SIMD width for uint64 */
    uint32_t arr9[9];    /* SIMD + 1 tail */
} all_types_t;

static const pack_field_t all_fields[] = {
    PACK_FIELD(all_types_t, u8,  PACK_U8),
    PACK_FIELD(all_types_t, u16, PACK_U16),
    PACK_FIELD(all_types_t, u32, PACK_U32),
    PACK_FIELD(all_types_t, u64, PACK_U64),
    PACK_FIELD(all_types_t, i8,  PACK_I8),
    PACK_FIELD(all_types_t, i16, PACK_I16),
    PACK_FIELD(all_types_t, i32, PACK_I32),
    PACK_FIELD(all_types_t, i64, PACK_I64),
    PACK_FIELD(all_types_t, f32, PACK_F32),
    PACK_FIELD(all_types_t, f64, PACK_F64),
    PACK_ARRAY_FIELD(all_types_t, arr8),
    PACK_ARRAY_FIELD(all_types_t, arr4),
    PACK_ARRAY_FIELD(all_types_t, arr9),
};

static const pack_schema_t all_schema = {
    .fields = all_fields,
    .field_count = 13,
    .struct_size = sizeof(all_types_t),
};

static int test_all_types_roundtrip(void)
{
    all_types_t src = {
        .u8 = 0xAB, .u16 = 0xCAFE, .u32 = 0xDEADBEEF, .u64 = 0x0123456789ABCDEFULL,
        .i8 = -42, .i16 = -1000, .i32 = -1000000, .i64 = -1000000000000LL,
        .f32 = 3.14159f, .f64 = 2.718281828459045,
        .arr8 = {1,2,3,4,5,6,7,8},
        .arr4 = {10,20,30,40},
        .arr9 = {100,200,300,400,500,600,700,800,900},
    };

    size_t wire_size = pack_serialized_size(&all_schema);
    uint8_t buf[512];
    size_t written;

    if (pack_serialize(&all_schema, &src, buf, sizeof(buf), &written) != PACK_OK) {
        fprintf(stderr, "FAIL all_types: serialize failed\n");
        return 1;
    }
    if (written != wire_size) {
        fprintf(stderr, "FAIL all_types: written=%zu, expected %zu\n", written, wire_size);
        return 1;
    }

    all_types_t dst;
    memset(&dst, 0, sizeof(dst));
    if (pack_deserialize(&all_schema, &dst, buf, written) != PACK_OK) {
        fprintf(stderr, "FAIL all_types: deserialize failed\n");
        return 1;
    }

    /* Compare field-by-field (memcmp would fail on padding bytes). */
    int ok = (src.u8 == dst.u8 && src.u16 == dst.u16 && src.u32 == dst.u32 &&
              src.u64 == dst.u64 && src.i8 == dst.i8 && src.i16 == dst.i16 &&
              src.i32 == dst.i32 && src.i64 == dst.i64 &&
              src.f32 == dst.f32 && src.f64 == dst.f64 &&
              memcmp(src.arr8, dst.arr8, sizeof(src.arr8)) == 0 &&
              memcmp(src.arr4, dst.arr4, sizeof(src.arr4)) == 0 &&
              memcmp(src.arr9, dst.arr9, sizeof(src.arr9)) == 0);

    if (!ok) {
        fprintf(stderr, "FAIL all_types: round-trip mismatch\n");
        return 1;
    }

    printf("PASS all_types: 13 fields (10 scalars + 3 arrays), %zu wire bytes, round-trip OK\n",
           wire_size);
    return 0;
}

static int test_fuzz_roundtrip(void)
{
    srand(12345);
    uint8_t buf[512];

    for (int round = 0; round < 10000; round++) {
        all_types_t src;
        /* Fill with random bytes. */
        for (size_t i = 0; i < sizeof(src); i++)
            ((uint8_t *)&src)[i] = rand() & 0xFF;

        size_t written;
        if (pack_serialize(&all_schema, &src, buf, sizeof(buf), &written) != PACK_OK) {
            fprintf(stderr, "FAIL fuzz: serialize round %d\n", round);
            return 1;
        }

        all_types_t dst;
        memset(&dst, 0, sizeof(dst));
        if (pack_deserialize(&all_schema, &dst, buf, written) != PACK_OK) {
            fprintf(stderr, "FAIL fuzz: deserialize round %d\n", round);
            return 1;
        }

        /* Compare field-by-field. Use memcmp for floats/doubles because
         * NaN != NaN even when bitwise-identical. */
        if (src.u8 != dst.u8 || src.u16 != dst.u16 || src.u32 != dst.u32 ||
            src.u64 != dst.u64 || src.i8 != dst.i8 || src.i16 != dst.i16 ||
            src.i32 != dst.i32 || src.i64 != dst.i64 ||
            memcmp(&src.f32, &dst.f32, sizeof(src.f32)) != 0 ||
            memcmp(&src.f64, &dst.f64, sizeof(src.f64)) != 0 ||
            memcmp(src.arr8, dst.arr8, sizeof(src.arr8)) != 0 ||
            memcmp(src.arr4, dst.arr4, sizeof(src.arr4)) != 0 ||
            memcmp(src.arr9, dst.arr9, sizeof(src.arr9)) != 0) {
            fprintf(stderr, "FAIL fuzz: mismatch round %d\n", round);
            return 1;
        }
    }

    printf("PASS fuzz_roundtrip: 10K random structs, all correct\n");
    return 0;
}

static int test_truncated_input(void)
{
    all_types_t src = {0};
    uint8_t buf[512];
    size_t written;
    pack_serialize(&all_schema, &src, buf, sizeof(buf), &written);

    all_types_t dst;
    /* Truncate to half the wire size. */
    if (pack_deserialize(&all_schema, &dst, buf, written / 2) != PACK_ERR_TRUNCATED) {
        fprintf(stderr, "FAIL truncated: expected TRUNCATED, got OK\n");
        return 1;
    }

    /* Zero-length input. */
    if (pack_deserialize(&all_schema, &dst, buf, 0) != PACK_ERR_TRUNCATED) {
        fprintf(stderr, "FAIL truncated: expected TRUNCATED for 0-length\n");
        return 1;
    }

    printf("PASS truncated_input: half-size and 0-length both return TRUNCATED\n");
    return 0;
}

static int test_buf_too_small(void)
{
    all_types_t src = {0};
    uint8_t buf[4]; /* way too small */
    size_t written;

    if (pack_serialize(&all_schema, &src, buf, sizeof(buf), &written) != PACK_ERR_BUF_TOO_SMALL) {
        fprintf(stderr, "FAIL buf_small: expected BUF_TOO_SMALL\n");
        return 1;
    }

    printf("PASS buf_too_small: 4-byte buffer rejected for %zu-byte schema\n",
           pack_serialized_size(&all_schema));
    return 0;
}

/* Latency: serialize+deserialize round-trip of a 64-byte struct. */
typedef struct {
    uint64_t v[8];
} pack64_t;

static const pack_field_t pack64_fields[] = {
    PACK_ARRAY_FIELD(pack64_t, v),
};
static const pack_schema_t pack64_schema = {
    .fields = pack64_fields,
    .field_count = 1,
    .struct_size = sizeof(pack64_t),
};

static int test_serialize_latency(void)
{
    const size_t N = 100000;
    uint64_t *samples = malloc(N * sizeof(uint64_t));
    if (!samples) { fprintf(stderr, "FAIL serialize_latency: malloc\n"); return 1; }

    pack64_t src = {0};
    uint8_t buf[64];
    size_t wire = pack_serialized_size(&pack64_schema);
    if (wire == 0 || wire > sizeof(buf)) {
        fprintf(stderr, "FAIL serialize_latency: wire=%zu\n", wire);
        free(samples);
        return 1;
    }

    for (size_t i = 0; i < N; i++) {
        src.v[0] = (uint64_t)i;
        pack64_t dst;
        size_t written;
        uint64_t t0 = latency_now_ns();
        pack_serialize(&pack64_schema, &src, buf, sizeof(buf), &written);
        pack_deserialize(&pack64_schema, &dst, buf, written);
        uint64_t t1 = latency_now_ns();
        samples[i] = t1 - t0;
    }

    uint64_t p50, p99, max;
    latency_stats(samples, N, &p50, &p99, &max);
    printf("=== latency ===\n");
    latency_print_ns("pack serialize+deserialize 64B", p50, p99, max, N);

    free(samples);
    return 0;
}

int main(void)
{
    int failures = 0;
    failures += test_all_types_roundtrip();
    failures += test_fuzz_roundtrip();
    failures += test_truncated_input();
    failures += test_buf_too_small();
    failures += test_serialize_latency();
    if (failures == 0) {
        printf("\nlibpack extreme: ALL PASS\n");
        return 0;
    }
    printf("\nlibpack extreme: %d FAILURE(S)\n", failures);
    return 1;
}
