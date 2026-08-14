#include <pack.h>
#include <immintrin.h>
#include <string.h>

/*
 * SIMD byte-swap primitives, multi-ISA variant set.
 *
 * The wire format is big-endian. On a little-endian host (every x86-64),
 * each multi-byte field must be byte-reversed on serialize and again on
 * deserialize. Array fields dominate the work, so the SIMD path matters.
 *
 * This file defines one function per (element width, ISA) combination.
 * Each ISA variant is compiled with __attribute__((target("..."))) which
 * enables the per-function ISA at the codegen level without forcing the
 * whole translation unit to require that ISA. The baseline CFLAGS carry
 * no -mavx2 / -mssse3 / -mavx512* flags; the public library symbol
 * works on any x86-64 baseline CPU because the variant that runs is
 * chosen at load time by the ifunc resolver in pack_dispatch.c.
 *
 * Variant set, per element width:
 *
 *   _scalar      - no target attribute; BSWAP r32 / r64 loop, always
 *                  available, baseline x86-64.
 *   _ssse3       - target("ssse3"); PSHUFB xmm, 16-byte per instruction
 *                  (4 uint32_t or 2 uint64_t). Available on every x86-64
 *                  since 2006 (Core 2 Merom).
 *   _avx2        - target("avx2"); VPSHUFB ymm, 32-byte per instruction
 *                  (8 uint32_t or 4 uint64_t). Haswell (2013) onward.
 *   _avx512      - target("avx512vbmi"); VPERMB zmm, 64-byte per
 *                  instruction (16 uint32_t or 8 uint64_t). Ice Lake
 *                  client (2019) / Cascade Lake server (2019) onward.
 *                  Unlike VPSHUFB ymm, VPERMB zmm can select any byte in
 *                  the 64-byte source, so the per-lane index restriction
 *                  that forces VPSHUFB's mask to repeat in 128-bit
 *                  lanes does not apply. This is the only difference
 *                  between the AVX2 and AVX-512 masks at the
 *                  element-size level; the per-element byte pattern
 *                  (reverse 4 or 8 bytes) is identical.
 *
 * Each non-scalar variant ends with a scalar tail loop that handles the
 * 1..N-1 trailing elements not consumed by the last SIMD instruction.
 * The scalar tail uses the same __builtin_bswap32/64 path as the
 * pure-scalar variant, so a partially-SIMD swap is bit-identical to a
 * full-scalar swap on the same input.
 *
 * Loads and stores use the unaligned intrinsics (_mm_loadu_si128,
 * _mm256_loadu_si256, _mm512_loadu_si512) because struct field
 * addresses and array element buffers are not guaranteed 16/32/64-byte
 * aligned. On Haswell and later the unaligned penalty is small for
 * L1-resident data; on AVX-512 hardware the same applies. The AVX-512
 * variants additionally check the array pointer for 64-byte alignment
 * at function entry and, when aligned, switch to _mm512_load_si512 /
 * _mm512_store_si512; on Skylake-X and later an aligned 512-bit load
 * avoids the unaligned-split penalty in the L1.
 *
 * The __m128i / __m256i / __m512i types carry __may_alias__ in GCC, so
 * accessing the caller's memory through these pointer types is
 * well-defined regardless of the original element type. The intermediate
 * uint32_t * and uint64_t * casts in the public-API wrappers are type
 * tokens only; the actual loads and stores go through the SIMD types
 * (or through unsigned char * in the scalar tail).
 *
 * The public entry points pack_bswap_uint32_array and
 * pack_bswap_uint64_array are NOT defined here; they are ifunc symbols
 * declared in pack_dispatch.c. pack_simd.c only defines the variants.
 * pack_bswap_float_array and pack_bswap_double_array are plain (non-
 * ifunc) wrappers that route through pack_bswap_uint32_array /
 * pack_bswap_uint64_array; the ifunc dispatch happens transparently.
 */

/* ---------------- byte-swap masks (one per element width per ISA) -------
 *
 * Stored as plain byte arrays (constant initializers in C11) and loaded
 * into vector registers at function entry. _mm*_set_epi8 does not
 * produce a C11 constant initializer under -pedantic, so the byte-array
 * form is used instead.
 *
 * For uint32 byte-swap, byte i of each 4-byte element selects byte (3-i)
 * within that element. For uint64 byte-swap, byte i of each 8-byte
 * element selects byte (7-i). The per-element pattern is identical
 * across SSSE3 / AVX2 / AVX-512; only the register width (16/32/64) and
 * the cross-lane behavior differ.
 *
 * VPSHUFB ymm and PSHUFB xmm mask the index with 0x0F before selecting
 * from the lane-local source, so the same 16-byte pattern repeats in
 * each 128-bit lane of the 32-byte ymm. VPERMB zmm masks with 0x3F and
 * can select any byte in the 64-byte zmm, so no lane repetition is
 * needed; the mask is just the per-element pattern repeated 16 times.
 */

static const unsigned char pack_bswap32_mask_ssse3[16] = {
     3,  2,  1,  0,  7,  6,  5,  4, 11, 10,  9,  8, 15, 14, 13, 12
};

static const unsigned char pack_bswap64_mask_ssse3[16] = {
     7,  6,  5,  4,  3,  2,  1,  0, 15, 14, 13, 12, 11, 10,  9,  8
};

static const unsigned char pack_bswap32_mask_avx2[32] = {
     3,  2,  1,  0,  7,  6,  5,  4, 11, 10,  9,  8, 15, 14, 13, 12,
    19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28
};

static const unsigned char pack_bswap64_mask_avx2[32] = {
     7,  6,  5,  4,  3,  2,  1,  0,
    15, 14, 13, 12, 11, 10,  9,  8,
    23, 22, 21, 20, 19, 18, 17, 16,
    31, 30, 29, 28, 27, 26, 25, 24
};

static const unsigned char pack_bswap32_mask_avx512[64] = {
     3,  2,  1,  0,  7,  6,  5,  4, 11, 10,  9,  8, 15, 14, 13, 12,
    19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28,
    35, 34, 33, 32, 39, 38, 37, 36, 43, 42, 41, 40, 47, 46, 45, 44,
    51, 50, 49, 48, 55, 54, 53, 52, 59, 58, 57, 56, 63, 62, 61, 60
};

static const unsigned char pack_bswap64_mask_avx512[64] = {
     7,  6,  5,  4,  3,  2,  1,  0, 15, 14, 13, 12, 11, 10,  9,  8,
    23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24,
    39, 38, 37, 36, 35, 34, 33, 32, 47, 46, 45, 44, 43, 42, 41, 40,
    55, 54, 53, 52, 51, 50, 49, 48, 63, 62, 61, 60, 59, 58, 57, 56
};

/* ---------------- scalar variant (no target attribute) ---------------- */

void pack_bswap_uint32_array_scalar(uint32_t *arr, size_t count)
{
    unsigned char *p = (unsigned char *)arr;
    for (size_t i = 0; i < count; i++) {
        uint32_t v;
        memcpy(&v, p + i * 4, sizeof(v));
        v = __builtin_bswap32(v);
        memcpy(p + i * 4, &v, sizeof(v));
    }
}

void pack_bswap_uint64_array_scalar(uint64_t *arr, size_t count)
{
    unsigned char *p = (unsigned char *)arr;
    for (size_t i = 0; i < count; i++) {
        uint64_t v;
        memcpy(&v, p + i * 8, sizeof(v));
        v = __builtin_bswap64(v);
        memcpy(p + i * 8, &v, sizeof(v));
    }
}

/* ---------------- SSSE3 variant (PSHUFB xmm, 16-byte) ---------------- */

__attribute__((target("ssse3")))
void pack_bswap_uint32_array_ssse3(uint32_t *arr, size_t count)
{
    unsigned char *p = (unsigned char *)arr;
    const __m128i mask =
        _mm_loadu_si128((const __m128i *)pack_bswap32_mask_ssse3);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i *)(p + i * 4));
        v = _mm_shuffle_epi8(v, mask);
        _mm_storeu_si128((__m128i *)(p + i * 4), v);
    }
    for (; i < count; i++) {
        uint32_t v;
        memcpy(&v, p + i * 4, sizeof(v));
        v = __builtin_bswap32(v);
        memcpy(p + i * 4, &v, sizeof(v));
    }
}

__attribute__((target("ssse3")))
void pack_bswap_uint64_array_ssse3(uint64_t *arr, size_t count)
{
    unsigned char *p = (unsigned char *)arr;
    const __m128i mask =
        _mm_loadu_si128((const __m128i *)pack_bswap64_mask_ssse3);

    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        __m128i v = _mm_loadu_si128((const __m128i *)(p + i * 8));
        v = _mm_shuffle_epi8(v, mask);
        _mm_storeu_si128((__m128i *)(p + i * 8), v);
    }
    for (; i < count; i++) {
        uint64_t v;
        memcpy(&v, p + i * 8, sizeof(v));
        v = __builtin_bswap64(v);
        memcpy(p + i * 8, &v, sizeof(v));
    }
}

/* ---------------- AVX2 variant (VPSHUFB ymm, 32-byte) ---------------- */

__attribute__((target("avx2")))
void pack_bswap_uint32_array_avx2(uint32_t *arr, size_t count)
{
    unsigned char *p = (unsigned char *)arr;
    const __m256i mask =
        _mm256_loadu_si256((const __m256i *)pack_bswap32_mask_avx2);

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(p + i * 4));
        v = _mm256_shuffle_epi8(v, mask);
        _mm256_storeu_si256((__m256i *)(p + i * 4), v);
    }
    for (; i < count; i++) {
        uint32_t v;
        memcpy(&v, p + i * 4, sizeof(v));
        v = __builtin_bswap32(v);
        memcpy(p + i * 4, &v, sizeof(v));
    }
}

__attribute__((target("avx2")))
void pack_bswap_uint64_array_avx2(uint64_t *arr, size_t count)
{
    unsigned char *p = (unsigned char *)arr;
    const __m256i mask =
        _mm256_loadu_si256((const __m256i *)pack_bswap64_mask_avx2);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(p + i * 8));
        v = _mm256_shuffle_epi8(v, mask);
        _mm256_storeu_si256((__m256i *)(p + i * 8), v);
    }
    for (; i < count; i++) {
        uint64_t v;
        memcpy(&v, p + i * 8, sizeof(v));
        v = __builtin_bswap64(v);
        memcpy(p + i * 8, &v, sizeof(v));
    }
}

/* ---------------- AVX-512 VBMI variant (VPERMB zmm, 64-byte) ---------
 *
 * Compiled in unconditionally. Selected at runtime only when the host
 * reports AVX-512F + AVX-512 VBMI in CPUID (Ice Lake client / Cascade
 * Lake server and later). On hosts without AVX-512 VBMI the resolver
 * never returns this variant, so the VPERMB zmm instruction is never
 * executed; the function body is present in the binary but unreachable
 * from the call site.
 *
 * VPERMB (vector byte permute) selects each destination byte from any
 * of the 64 source bytes, using a 6-bit index (0x3F mask) per byte.
 * This is a true cross-lane permute at the 64-byte granularity, unlike
 * VPSHUFB ymm which is two independent 16-byte lane-local permutes.
 * The throughput win over AVX2 is 2x (64 bytes per instruction vs 32).
 */

__attribute__((target("avx512vbmi")))
void pack_bswap_uint32_array_avx512(uint32_t *arr, size_t count)
{
    unsigned char *p = (unsigned char *)arr;
    const __m512i mask =
        _mm512_loadu_si512((const void *)pack_bswap32_mask_avx512);

    /* Aligned fast path: if the array starts on a 64-byte boundary,
     * every subsequent 64-byte block (16 uint32_t per VPERMB zmm) is
     * also 64-byte aligned, so _mm512_load_si512 / _mm512_store_si512
     * are safe and faster than the unaligned forms on Skylake-X.
     * The check is a single AND of the low 6 bits; the cost is one
     * branch at function entry, taken or not-taken once per call. */
    int aligned = ((uintptr_t)p % 64 == 0);

    size_t i = 0;
    if (aligned) {
        for (; i + 16 <= count; i += 16) {
            __m512i v = _mm512_load_si512((const void *)(p + i * 4));
            v = _mm512_permutexvar_epi8(mask, v);
            _mm512_store_si512((void *)(p + i * 4), v);
        }
    } else {
        for (; i + 16 <= count; i += 16) {
            __m512i v = _mm512_loadu_si512((const void *)(p + i * 4));
            v = _mm512_permutexvar_epi8(mask, v);
            _mm512_storeu_si512((void *)(p + i * 4), v);
        }
    }
    /* Tail: handle the remaining 1..15 elements. There is no AVX-512
     * mask-register load path here because the tail is short and the
     * scalar BSWAP loop is already optimal for a handful of elements. */
    for (; i < count; i++) {
        uint32_t v;
        memcpy(&v, p + i * 4, sizeof(v));
        v = __builtin_bswap32(v);
        memcpy(p + i * 4, &v, sizeof(v));
    }
}

__attribute__((target("avx512vbmi")))
void pack_bswap_uint64_array_avx512(uint64_t *arr, size_t count)
{
    unsigned char *p = (unsigned char *)arr;
    const __m512i mask =
        _mm512_loadu_si512((const void *)pack_bswap64_mask_avx512);

    /* Aligned fast path: see pack_bswap_uint32_array_avx512. The same
     * 64-byte alignment check applies; each VPERMB zmm consumes 8
     * uint64_t = 64 bytes. */
    int aligned = ((uintptr_t)p % 64 == 0);

    size_t i = 0;
    if (aligned) {
        for (; i + 8 <= count; i += 8) {
            __m512i v = _mm512_load_si512((const void *)(p + i * 8));
            v = _mm512_permutexvar_epi8(mask, v);
            _mm512_store_si512((void *)(p + i * 8), v);
        }
    } else {
        for (; i + 8 <= count; i += 8) {
            __m512i v = _mm512_loadu_si512((const void *)(p + i * 8));
            v = _mm512_permutexvar_epi8(mask, v);
            _mm512_storeu_si512((void *)(p + i * 8), v);
        }
    }
    for (; i < count; i++) {
        uint64_t v;
        memcpy(&v, p + i * 8, sizeof(v));
        v = __builtin_bswap64(v);
        memcpy(p + i * 8, &v, sizeof(v));
    }
}

/* ---------------- float / double wrappers ----------------
 *
 * IEEE-754 float / double byte order matches uint32_t / uint64_t byte
 * order on every x86-64 and every little-endian AArch64, so swapping
 * the underlying 4 or 8 bytes is the correct big-endian wire conversion.
 * The call routes through the ifunc-dispatched public symbol, so the
 * variant actually executed is whatever the resolver picked at load
 * time (avx512vbmi / avx2 / ssse3 / scalar) -- no extra per-call branch.
 *
 * The unsigned char * view through which the swap routine accesses the
 * memory may alias any object type per C11 6.5p7, so passing a float *
 * or double * through a uint32_t * or uint64_t * cast here is a type
 * token only; the swap routine never dereferences the integer pointer
 * directly (the SIMD path goes through __m256i *, which carries
 * __may_alias__ in GCC, and the scalar path goes through unsigned char *).
 */

void pack_bswap_float_array(float *arr, size_t count)
{
    pack_bswap_uint32_array((uint32_t *)(void *)arr, count);
}

void pack_bswap_double_array(double *arr, size_t count)
{
    pack_bswap_uint64_array((uint64_t *)(void *)arr, count);
}
