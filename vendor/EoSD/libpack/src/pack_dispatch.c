#include <pack.h>

/*
 * Runtime CPU dispatch via a function pointer initialized once in a
 * constructor. This replaces the ifunc mechanism which caused crashes
 * on some glibc/ld combinations where the IRELATIVE relocation handler
 * runs before __builtin_cpu_supports is ready.
 *
 * The constructor runs after libc init, so __builtin_cpu_supports is
 * safe. The dispatch cost is one indirect call through a static
 * function pointer, same overhead as a PLT call.
 *
 * Selection order (most preferred first):
 *
 *   avx512vbmi  - VPERMB zmm, 64-byte parallel byte permute
 *   avx2        - VPSHUFB ymm, 32-byte parallel byte shuffle
 *   ssse3       - PSHUFB xmm, 16-byte parallel byte shuffle
 *   scalar      - BSWAP r32 / r64 loop, baseline x86-64
 */

typedef void (*pack_bswap32_fn)(uint32_t *, size_t);
typedef void (*pack_bswap64_fn)(uint64_t *, size_t);

void pack_bswap_uint32_array_avx512(uint32_t *, size_t);
void pack_bswap_uint32_array_avx2(uint32_t *, size_t);
void pack_bswap_uint32_array_ssse3(uint32_t *, size_t);
void pack_bswap_uint32_array_scalar(uint32_t *, size_t);

void pack_bswap_uint64_array_avx512(uint64_t *, size_t);
void pack_bswap_uint64_array_avx2(uint64_t *, size_t);
void pack_bswap_uint64_array_ssse3(uint64_t *, size_t);
void pack_bswap_uint64_array_scalar(uint64_t *, size_t);

static pack_bswap32_fn g_bswap32 = pack_bswap_uint32_array_scalar;
static pack_bswap64_fn g_bswap64 = pack_bswap_uint64_array_scalar;

__attribute__((constructor))
static void pack_init_dispatch(void)
{
    if (__builtin_cpu_supports("avx512vbmi")) {
        g_bswap32 = pack_bswap_uint32_array_avx512;
        g_bswap64 = pack_bswap_uint64_array_avx512;
    } else if (__builtin_cpu_supports("avx2")) {
        g_bswap32 = pack_bswap_uint32_array_avx2;
        g_bswap64 = pack_bswap_uint64_array_avx2;
    } else if (__builtin_cpu_supports("ssse3")) {
        g_bswap32 = pack_bswap_uint32_array_ssse3;
        g_bswap64 = pack_bswap_uint64_array_ssse3;
    }
}

void pack_bswap_uint32_array(uint32_t *arr, size_t count)
{
    g_bswap32(arr, count);
}

void pack_bswap_uint64_array(uint64_t *arr, size_t count)
{
    g_bswap64(arr, count);
}
