#include <pack.h>
#include <stdlib.h>
#include <string.h>

/*
 * Schema-driven (de)serialization.
 *
 * The wire format is a tight concatenation of field bytes, in big-endian
 * byte order. Padding between fields and at the struct tail is not
 * serialized; the loop walks the schema's (offset, size) pairs and emits
 * only the field's own bytes. On a little-endian host (every x86-64) the
 * emitted bytes are byte-swapped in place after the memcpy; on a
 * big-endian host the wire format is already native and no swap is
 * needed.
 *
 * Compile-time endianness detection uses the GCC/Clang predefined macros
 * __BYTE_ORDER__, __ORDER_LITTLE_ENDIAN__, __ORDER_BIG_ENDIAN__. C23's
 * <stdbit.h> is C23, not C11, and is not used.
 *
 * Single struct members use the scalar __builtin_bswap32/64 path (one
 * BSWAP instruction on x86-64). Array fields route through the SIMD
 * helpers in pack_simd.c, which are ifunc-dispatched at runtime: the
 * resolver in pack_dispatch.c picks avx512vbmi (VPERMB zmm) > avx2
 * (VPSHUFB ymm, 8 uint32_t / 4 uint64_t per instruction) > ssse3
 * (PSHUFB xmm) > scalar (BSWAP r32/r64 loop) based on the host CPU.
 * The split follows DESIGN.md: scalar for singles, SIMD for arrays.
 *
 * v0.2 adds variable-length field types (PACK_STRING, PACK_BYTES,
 * PACK_VAR_ARRAY). Each is encoded as a 4-byte big-endian byte count
 * followed by the data bytes. For PACK_VAR_ARRAY the data bytes are
 * element_count * element_size, and each element is byte-swapped using
 * the same path as PACK_ARRAY. The version prefix (when non-zero) is
 * itself a 4-byte big-endian uint32, encoded with the same helpers.
 *
 * All byte-swapped memory is reached through unsigned char *, which may
 * alias any object type per C11 6.5p7. The intermediate uint32_t * and
 * uint64_t * casts on the way into the SIMD helpers are type tokens
 * only; the actual loads and stores go through unsigned char * (and
 * through __m256i *, which carries __may_alias__ in GCC).
 */

#if !defined(__BYTE_ORDER__)
#error "unrecognized endianness; __BYTE_ORDER__ not defined"
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define PACK_HOST_LITTLE_ENDIAN 1
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define PACK_HOST_LITTLE_ENDIAN 0
#else
#error "unrecognized endianness; __BYTE_ORDER__ is neither little nor big"
#endif

/* -------- endianness helpers -------- */

/*
 * Write uint32 in big-endian wire form. Explicit byte shifts so the
 * result is host-endian independent; gcc -O2 folds this to a single
 * BSWAP reg32 on x86-64.
 */
static void pack_put_u32_be(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v >> 24);
    p[1] = (unsigned char)(v >> 16);
    p[2] = (unsigned char)(v >>  8);
    p[3] = (unsigned char)(v);
}

static uint32_t pack_get_u32_be(const unsigned char *p)
{
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] <<  8) |
            (uint32_t)p[3];
}

/* -------- field type helpers -------- */

static int pack_is_var_type(pack_field_type_t t)
{
    return t == PACK_STRING || t == PACK_BYTES || t == PACK_VAR_ARRAY;
}

int pack_schema_has_variable(const pack_schema_t *schema)
{
    if (!schema) {
        return 0;
    }
    for (size_t i = 0; i < schema->field_count; i++) {
        if (pack_is_var_type(schema->fields[i].type)) {
            return 1;
        }
    }
    return 0;
}

/*
 * Decode one fixed-size field's element size and element count from its
 * descriptor. Scalar fields have count = 1; fixed-size array fields use
 * the recorded count. Variable-length fields are not handled here.
 */
static void pack_field_shape(const pack_field_t *f, size_t *elem_size,
                             size_t *count)
{
    if (f->type == PACK_ARRAY) {
        *elem_size = f->size;
        *count = f->array_count;
    } else {
        *elem_size = f->size;
        *count = 1;
    }
}

#if PACK_HOST_LITTLE_ENDIAN

/*
 * Byte-swap `count` elements of `elem_size` bytes each, in place, in the
 * buffer `buf`. elem_size picks the swap width: 1 byte is a no-op, 2 uses
 * __builtin_bswap16 (ROL r16,8 on x86), 4 uses __builtin_bswap32 (BSWAP
 * r32) for the single-element case or the SIMD path for arrays, 8 uses
 * __builtin_bswap64 (BSWAP r64) for the single-element case or the SIMD
 * path for arrays. Other element sizes are left alone (the schema should
 * not produce any).
 */
static void pack_swap_region(unsigned char *buf, size_t elem_size,
                             size_t count)
{
    if (count == 0) {
        return;
    }
    switch (elem_size) {
    case 1:
        break;
    case 2:
        for (size_t i = 0; i < count; i++) {
            uint16_t v;
            memcpy(&v, buf + i * 2, sizeof(v));
            v = __builtin_bswap16(v);
            memcpy(buf + i * 2, &v, sizeof(v));
        }
        break;
    case 4:
        if (count <= 4) {
            /*
             * Small-count fast path: for 1..4 uint32_t (4..16 bytes),
             * scalar __builtin_bswap32 is faster than routing through
             * the ifunc-dispatched SIMD helper. The SIMD variants all
             * fall through to a scalar tail for counts below their
             * register width (16 for avx512, 8 for avx2, 4 for ssse3),
             * so the only thing the SIMD call adds for these small
             * counts is the indirect call + mask load + alignment
             * check + loop-setup overhead. A tight scalar loop wins.
             */
            for (size_t i = 0; i < count; i++) {
                uint32_t v;
                memcpy(&v, buf + i * 4, sizeof(v));
                v = __builtin_bswap32(v);
                memcpy(buf + i * 4, &v, sizeof(v));
            }
        } else {
            pack_bswap_uint32_array((uint32_t *)(void *)buf, count);
        }
        break;
    case 8:
        if (count <= 2) {
            /*
             * Small-count fast path: 1..2 uint64_t (8..16 bytes).
             * Same reasoning as the uint32 case: scalar BSWAP wins
             * over the ifunc dispatch + SIMD setup for very short
             * runs.
             */
            for (size_t i = 0; i < count; i++) {
                uint64_t v;
                memcpy(&v, buf + i * 8, sizeof(v));
                v = __builtin_bswap64(v);
                memcpy(buf + i * 8, &v, sizeof(v));
            }
        } else {
            pack_bswap_uint64_array((uint64_t *)(void *)buf, count);
        }
        break;
    default:
        break;
    }
}

#endif /* PACK_HOST_LITTLE_ENDIAN */

/* -------- size computation -------- */

size_t pack_serialized_size(const pack_schema_t *schema)
{
    if (!schema) {
        return 0;
    }
    /*
     * Variable-length schemas have no fixed size; return 0 so a caller
     * that forgets to use pack_serialized_size_var fails loudly with
     * PACK_ERR_BUF_TOO_SMALL at serialize time rather than silently
     * underallocating. The version prefix (when non-zero) is still
     * counted for fixed-only schemas so pack_serialized_size matches
     * pack_serialized_size_var for those.
     */
    if (pack_schema_has_variable(schema)) {
        return 0;
    }
    size_t total = 0;
    if (schema->schema_version != 0) {
        total += 4;
    }
    for (size_t i = 0; i < schema->field_count; i++) {
        const pack_field_t *f = &schema->fields[i];
        if (f->type == PACK_ARRAY) {
            total += f->size * f->array_count;
        } else {
            total += f->size;
        }
    }
    return total;
}

size_t pack_serialized_size_var(const pack_schema_t *schema,
                                const void *struct_ptr)
{
    if (!schema || !struct_ptr) {
        return 0;
    }
    size_t total = 0;
    if (schema->schema_version != 0) {
        total += 4;
    }
    const unsigned char *base = (const unsigned char *)struct_ptr;
    for (size_t i = 0; i < schema->field_count; i++) {
        const pack_field_t *f = &schema->fields[i];
        if (pack_is_var_type(f->type)) {
            size_t len;
            memcpy(&len, base + f->len_offset, sizeof(len));
            size_t byte_count;
            if (f->type == PACK_VAR_ARRAY) {
                /*
                 * len is element count; size is element size. The
                 * multiply can wrap on 64-bit hosts (e.g. len = 2^62+1,
                 * size = 4 wraps to 4), bypassing the byte_count >
                 * UINT32_MAX check in pack_serialize_into. Reject the
                 * overflow here by returning 0, which signals "no size
                 * available" to pack_serialize / pack_serialize_alloc
                 * and forces the actual serializer into its BAD_LEN
                 * path.
                 */
                if (f->size != 0 && len > UINT32_MAX / f->size) {
                    return 0;
                }
                byte_count = len * f->size;
            } else {
                /* PACK_STRING / PACK_BYTES: len is already byte count */
                byte_count = len;
            }
            total += 4 + byte_count;
        } else if (f->type == PACK_ARRAY) {
            total += f->size * f->array_count;
        } else {
            total += f->size;
        }
    }
    return total;
}

/* -------- allocator plumbing -------- */

static void *pack_alloc(const pack_allocator_t *allocator, size_t size)
{
    if (!allocator || !allocator->alloc) {
        return malloc(size);
    }
    return allocator->alloc(size, allocator->user_data);
}

static void pack_free(const pack_allocator_t *allocator, void *ptr,
                      size_t size)
{
    if (!ptr) {
        return;
    }
    if (!allocator || !allocator->free) {
        free(ptr);
        return;
    }
    allocator->free(ptr, size, allocator->user_data);
}

/* -------- serialize (caller-supplied buffer) -------- */

/*
 * Core serializer. Writes into out_buf starting at *io_pos. Returns
 * PACK_OK or a negative error code; advances *io_pos on success. The
 * version prefix and bounds check are done by the caller.
 */
static int pack_serialize_into(const pack_schema_t *schema,
                               const void *struct_ptr,
                               unsigned char *out, size_t buf_size,
                               size_t *io_pos)
{
    const unsigned char *src = (const unsigned char *)struct_ptr;
    size_t pos = *io_pos;

    for (size_t i = 0; i < schema->field_count; i++) {
        const pack_field_t *f = &schema->fields[i];

        if (pack_is_var_type(f->type)) {
            size_t len;
            memcpy(&len, src + f->len_offset, sizeof(len));

            size_t byte_count;
            if (f->type == PACK_VAR_ARRAY) {
                /*
                 * Multiply-overflow guard. Without this, len * f->size
                 * can wrap on 64-bit hosts to a value <= UINT32_MAX,
                 * bypassing the byte_count > UINT32_MAX check below and
                 * then handing the unwrapped len to pack_swap_region,
                 * which walks off the end of the output buffer.
                 */
                if (f->size != 0 && len > UINT32_MAX / f->size) {
                    return PACK_ERR_BAD_LEN;
                }
                byte_count = len * f->size;
            } else {
                byte_count = len;
            }
            if (byte_count > 0xFFFFFFFFu) {
                return PACK_ERR_BAD_LEN;
            }
            const void *data_ptr;
            memcpy(&data_ptr, src + f->offset, sizeof(data_ptr));
            if (byte_count > 0 && data_ptr == NULL) {
                return PACK_ERR_INVALID;
            }

            if (buf_size - pos < 4 + byte_count) {
                return PACK_ERR_BUF_TOO_SMALL;
            }
            pack_put_u32_be(out + pos, (uint32_t)byte_count);
            pos += 4;
            if (byte_count > 0) {
                memcpy(out + pos, data_ptr, byte_count);
#if PACK_HOST_LITTLE_ENDIAN
                if (f->type == PACK_VAR_ARRAY && f->size > 1) {
                    pack_swap_region(out + pos, f->size, len);
                }
#endif
                pos += byte_count;
            }
        } else {
            size_t elem_size, count;
            pack_field_shape(f, &elem_size, &count);
            size_t bytes = elem_size * count;
            if (buf_size - pos < bytes) {
                return PACK_ERR_BUF_TOO_SMALL;
            }
            memcpy(out + pos, src + f->offset, bytes);
#if PACK_HOST_LITTLE_ENDIAN
            pack_swap_region(out + pos, elem_size, count);
#endif
            pos += bytes;
        }
    }

    *io_pos = pos;
    return PACK_OK;
}

int pack_serialize(const pack_schema_t *schema, const void *struct_ptr,
                   void *out_buf, size_t buf_size, size_t *out_written)
{
    if (!schema || !struct_ptr || !out_buf) {
        return PACK_ERR_INVALID;
    }
    if (!schema->fields && schema->field_count > 0) {
        return PACK_ERR_INVALID;
    }

    /*
     * Skip the pack_serialized_size_var pre-walk. pack_serialize_into
     * already bounds-checks every field against (buf_size - pos), so the
     * pre-walk is redundant work: it walks the schema a second time just
     * to compute the same total that pack_serialize_into will produce
     * incrementally. The only pre-check we still need is room for the
     * 4-byte version prefix (when present); pack_serialize_into does
     * not see the prefix because it is written here, before the call.
     *
     * On error (PACK_ERR_BUF_TOO_SMALL / PACK_ERR_BAD_LEN), the version
     * prefix may already have been written. The API contract is that
     * *out_written is only written on success, and the caller must not
     * inspect out_buf on a non-OK return, so a partial prefix write is
     * safe. For schemas with no version prefix, no write occurs before
     * pack_serialize_into's first per-field bounds check, so the
     * "reject without writing" property is preserved exactly.
     *
     * Measured impact: on a 64-byte single-array struct, this trims
     * one full schema-walk function call (and its pack_is_var_type
     * per-field comparisons) off every serialize, bringing the
     * serialize path within ~1 ns of the deserialize path (which
     * already had no pre-walk). The ifunc-resolved SIMD path itself
     * is unchanged.
     */
    if (schema->schema_version != 0 && buf_size < 4) {
        return PACK_ERR_BUF_TOO_SMALL;
    }

    unsigned char *out = (unsigned char *)out_buf;
    size_t pos = 0;

    if (schema->schema_version != 0) {
        pack_put_u32_be(out, schema->schema_version);
        pos += 4;
    }

    int rc = pack_serialize_into(schema, struct_ptr, out, buf_size, &pos);
    if (rc != PACK_OK) {
        return rc;
    }

    if (out_written) {
        *out_written = pos;
    }
    return PACK_OK;
}

/* -------- deserialize (caller-supplied struct) -------- */

/*
 * Core deserializer. Reads from in_buf starting at *io_pos, writes into
 * struct_ptr's fields. Returns PACK_OK or a negative error code;
 * advances *io_pos on success. The version-prefix check (if any) is
 * done by the caller.
 */
static int pack_deserialize_into(const pack_schema_t *schema,
                                 void *struct_ptr,
                                 const unsigned char *in, size_t buf_size,
                                 size_t *io_pos)
{
    unsigned char *dst = (unsigned char *)struct_ptr;
    size_t pos = *io_pos;

    for (size_t i = 0; i < schema->field_count; i++) {
        const pack_field_t *f = &schema->fields[i];

        if (pack_is_var_type(f->type)) {
            if (buf_size - pos < 4) {
                return PACK_ERR_TRUNCATED;
            }
            uint32_t byte_count = pack_get_u32_be(in + pos);
            pos += 4;
            if (buf_size - pos < byte_count) {
                return PACK_ERR_TRUNCATED;
            }

            size_t len_to_store;
            void *data_ptr;
            memcpy(&data_ptr, dst + f->offset, sizeof(data_ptr));

            if (f->type == PACK_VAR_ARRAY) {
                if (f->size == 0 ||
                    byte_count % f->size != 0) {
                    return PACK_ERR_BAD_LEN;
                }
                len_to_store = byte_count / f->size;
            } else {
                len_to_store = byte_count;
            }

            if (byte_count > 0) {
                if (data_ptr == NULL) {
                    /* Caller gave us a NULL data pointer but the wire
                     * has bytes; cannot write them. */
                    return PACK_ERR_INVALID;
                }
                memcpy(data_ptr, in + pos, byte_count);
#if PACK_HOST_LITTLE_ENDIAN
                if (f->type == PACK_VAR_ARRAY && f->size > 1) {
                    pack_swap_region((unsigned char *)data_ptr,
                                     f->size, len_to_store);
                }
#endif
                pos += byte_count;
            }
            memcpy(dst + f->len_offset, &len_to_store,
                   sizeof(len_to_store));
        } else {
            size_t elem_size, count;
            pack_field_shape(f, &elem_size, &count);
            size_t bytes = elem_size * count;
            if (buf_size - pos < bytes) {
                return PACK_ERR_TRUNCATED;
            }
            memcpy(dst + f->offset, in + pos, bytes);
#if PACK_HOST_LITTLE_ENDIAN
            pack_swap_region(dst + f->offset, elem_size, count);
#endif
            pos += bytes;
        }
    }

    *io_pos = pos;
    return PACK_OK;
}

int pack_deserialize(const pack_schema_t *schema, void *struct_ptr,
                     const void *in_buf, size_t buf_size)
{
    if (!schema || !struct_ptr || !in_buf) {
        return PACK_ERR_INVALID;
    }
    if (!schema->fields && schema->field_count > 0) {
        return PACK_ERR_INVALID;
    }

    const unsigned char *in = (const unsigned char *)in_buf;
    size_t pos = 0;

    if (schema->schema_version != 0) {
        if (buf_size - pos < 4) {
            return PACK_ERR_TRUNCATED;
        }
        uint32_t wire_version = pack_get_u32_be(in + pos);
        pos += 4;
        if (wire_version != schema->schema_version) {
            return PACK_ERR_SCHEMA_VERSION;
        }
    }

    return pack_deserialize_into(schema, struct_ptr, in, buf_size, &pos);
}

/* -------- serialize_alloc -------- */

int pack_serialize_alloc(const pack_schema_t *schema,
                         const void *struct_ptr,
                         void **out_buf, size_t *out_size,
                         const pack_allocator_t *allocator)
{
    if (!schema || !struct_ptr || !out_buf || !out_size) {
        return PACK_ERR_INVALID;
    }
    if (!schema->fields && schema->field_count > 0) {
        return PACK_ERR_INVALID;
    }

    size_t total = pack_serialized_size_var(schema, struct_ptr);

    unsigned char *buf = (unsigned char *)pack_alloc(allocator, total);
    if (!buf && total > 0) {
        return PACK_ERR_ALLOC;
    }

    size_t pos = 0;
    if (schema->schema_version != 0) {
        pack_put_u32_be(buf, schema->schema_version);
        pos += 4;
    }

    int rc = pack_serialize_into(schema, struct_ptr, buf, total, &pos);
    if (rc != PACK_OK) {
        pack_free(allocator, buf, total);
        return rc;
    }

    *out_buf = buf;
    *out_size = pos;
    return PACK_OK;
}

/* -------- deserialize_alloc -------- */

int pack_deserialize_alloc(const pack_schema_t *schema,
                           const void *in_buf, size_t in_size,
                           void **out_struct,
                           const pack_allocator_t *allocator)
{
    if (!schema || !in_buf || !out_struct) {
        return PACK_ERR_INVALID;
    }
    if (!schema->fields && schema->field_count > 0) {
        return PACK_ERR_INVALID;
    }
    if (schema->struct_size == 0) {
        return PACK_ERR_INVALID;
    }

    *out_struct = NULL;

    unsigned char *st = (unsigned char *)pack_alloc(allocator,
                                                    schema->struct_size);
    if (!st) {
        return PACK_ERR_ALLOC;
    }
    memset(st, 0, schema->struct_size);

    const unsigned char *in = (const unsigned char *)in_buf;
    size_t pos = 0;

    if (schema->schema_version != 0) {
        if (in_size - pos < 4) {
            pack_free(allocator, st, schema->struct_size);
            return PACK_ERR_TRUNCATED;
        }
        uint32_t wire_version = pack_get_u32_be(in + pos);
        pos += 4;
        if (wire_version != schema->schema_version) {
            pack_free(allocator, st, schema->struct_size);
            return PACK_ERR_SCHEMA_VERSION;
        }
    }

    /*
     * Walk the fields. For fixed-size fields, use the existing path.
     * For variable-length fields, allocate the data buffer here so we
     * can populate the struct's pointer field directly. On any failure
     * mid-walk, roll back via pack_free_struct.
     */
    for (size_t i = 0; i < schema->field_count; i++) {
        const pack_field_t *f = &schema->fields[i];

        if (pack_is_var_type(f->type)) {
            if (in_size - pos < 4) {
                pack_free_struct(schema, st, allocator);
                return PACK_ERR_TRUNCATED;
            }
            uint32_t byte_count = pack_get_u32_be(in + pos);
            pos += 4;
            if (in_size - pos < byte_count) {
                pack_free_struct(schema, st, allocator);
                return PACK_ERR_TRUNCATED;
            }

            size_t len_to_store;
            if (f->type == PACK_VAR_ARRAY) {
                if (f->size == 0 || byte_count % f->size != 0) {
                    pack_free_struct(schema, st, allocator);
                    return PACK_ERR_BAD_LEN;
                }
                len_to_store = byte_count / f->size;
            } else {
                len_to_store = byte_count;
            }

            unsigned char *data_buf = NULL;
            if (byte_count > 0) {
                data_buf = (unsigned char *)pack_alloc(allocator,
                                                       byte_count);
                if (!data_buf) {
                    pack_free_struct(schema, st, allocator);
                    return PACK_ERR_ALLOC;
                }
                memcpy(data_buf, in + pos, byte_count);
#if PACK_HOST_LITTLE_ENDIAN
                if (f->type == PACK_VAR_ARRAY && f->size > 1) {
                    pack_swap_region(data_buf, f->size, len_to_store);
                }
#endif
                pos += byte_count;
            }
            memcpy(st + f->offset, &data_buf, sizeof(data_buf));
            memcpy(st + f->len_offset, &len_to_store,
                   sizeof(len_to_store));
        } else {
            size_t elem_size, count;
            pack_field_shape(f, &elem_size, &count);
            size_t bytes = elem_size * count;
            if (in_size - pos < bytes) {
                pack_free_struct(schema, st, allocator);
                return PACK_ERR_TRUNCATED;
            }
            memcpy(st + f->offset, in + pos, bytes);
#if PACK_HOST_LITTLE_ENDIAN
            pack_swap_region(st + f->offset, elem_size, count);
#endif
            pos += bytes;
        }
    }

    *out_struct = st;
    return PACK_OK;
}

/* -------- pack_free_struct -------- */

int pack_free_struct(const pack_schema_t *schema, void *struct_ptr,
                     const pack_allocator_t *allocator)
{
    if (!struct_ptr) {
        return PACK_OK;
    }
    if (!schema) {
        return PACK_ERR_INVALID;
    }

    unsigned char *st = (unsigned char *)struct_ptr;
    for (size_t i = 0; i < schema->field_count; i++) {
        const pack_field_t *f = &schema->fields[i];
        if (!pack_is_var_type(f->type)) {
            continue;
        }
        void *data_ptr;
        memcpy(&data_ptr, st + f->offset, sizeof(data_ptr));
        if (!data_ptr) {
            continue;
        }
        size_t len;
        memcpy(&len, st + f->len_offset, sizeof(len));
        size_t byte_count;
        if (f->type == PACK_VAR_ARRAY) {
            /*
             * Multiply-overflow guard. The deserializer stores
             * len <= UINT32_MAX / f->size, so in normal operation no
             * overflow is possible; this guards against caller-corrupted
             * len values that would otherwise wrap byte_count and pass a
             * wrong size hint to the allocator's free. libc free ignores
             * the size argument; a tracking allocator should look up the
             * original size by pointer. Use SIZE_MAX as the "size
             * unknown" sentinel because that is the most obviously
             * wrong value a strict allocator could detect.
             */
            if (f->size != 0 && len > UINT32_MAX / f->size) {
                byte_count = SIZE_MAX;
            } else {
                byte_count = len * f->size;
            }
        } else {
            byte_count = len;
        }
        pack_free(allocator, data_ptr, byte_count);
        /* NULL out the pointer so a double-free is a no-op. */
        void *null_ptr = NULL;
        memcpy(st + f->offset, &null_ptr, sizeof(null_ptr));
    }

    pack_free(allocator, st, schema->struct_size);
    return PACK_OK;
}
