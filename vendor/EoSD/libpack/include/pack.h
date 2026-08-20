#ifndef PACK_H
#define PACK_H

#include <stddef.h>
#include <stdint.h>

/*
 * libpack: compile-time-described struct (de)serialization with SIMD
 * endianness conversion.
 *
 * Wire format is big-endian, flat (no TLV). Padding bytes between fields
 * and at the struct tail are NOT serialized. Output is a tight
 * concatenation of field bytes.
 *
 * v0.1: fixed-size scalars and fixed-size arrays only. Schema-described
 * size, so pack_serialized_size(schema) is a pure function of schema.
 *
 * v0.2: adds variable-length fields (PACK_STRING, PACK_BYTES,
 * PACK_VAR_ARRAY) with a 4-byte big-endian length prefix, an optional
 * pluggable allocator (pack_allocator_t) for serialize-into-heap /
 * deserialize-from-heap paths, and optional schema versioning
 * (schema_version prefix when non-zero). The v0.1 caller-supplied-buffer
 * entry points remain and continue to work for both fixed-only and
 * variable-length schemas.
 *
 * Thread-safety: pack_serialize, pack_deserialize,
 * pack_serialized_size, pack_serialized_size_var, pack_serialize_alloc,
 * pack_deserialize_alloc, and pack_free_struct are pure functions of
 * their arguments and may be called concurrently from multiple threads
 * on different buffers. The SIMD bswap helpers operate in-place on
 * caller-owned memory and have the same guarantee.
 */

typedef enum {
    PACK_U8,
    PACK_U16,
    PACK_U32,
    PACK_U64,
    PACK_I8,
    PACK_I16,
    PACK_I32,
    PACK_I64,
    PACK_F32,
    PACK_F64,
    PACK_ARRAY,
    /*
     * v0.2 variable-length field types. Each is serialized as a 4-byte
     * big-endian byte count followed by the data bytes. For
     * PACK_VAR_ARRAY the byte count is element_count * element_size.
     */
    PACK_STRING,     /* UTF-8 bytes; field is char *, len_field is bytes */
    PACK_BYTES,      /* raw bytes;   field is uint8_t *, len_field is bytes */
    PACK_VAR_ARRAY   /* fixed-size elements; field is T *, len_field is count */
} pack_field_type_t;

typedef struct {
    const char *name;
    pack_field_type_t type;
    size_t offset;        /* offsetof(Struct, field) */
    size_t size;          /* sizeof(field) for scalars, element size for
                           * arrays and PACK_VAR_ARRAY (1 for STRING/BYTES) */
    size_t array_count;   /* 0 for scalar fields, element count for fixed
                           * arrays, unused for variable-length fields */
    size_t len_offset;    /* offsetof(Struct, len_field) for variable-length
                           * fields; 0 otherwise (unused) */
} pack_field_t;

/*
 * PACK_FIELD builds a scalar field descriptor. offsetof and sizeof are
 * both integer constant expressions in C11, so the result is a constant
 * initializer; a typo'd field name is a hard compile error. Applying
 * this macro to a bit-field member is itself a compile error
 * (sizeof/offsetof are not applicable to bit-fields, C11 6.5.3.4p1 and
 * 7.19p3) -- this is the desired safety property, not a limitation to
 * work around.
 */
#define PACK_FIELD(Struct, field, ftype)                                     \
    {                                                                        \
        #field, (ftype), offsetof(Struct, field),                            \
            sizeof(((Struct *)0)->field), 0, 0                               \
    }

/*
 * PACK_ARRAY_FIELD builds a descriptor for a fixed-size array field.
 * The element size and element count are both derived from the array
 * declaration itself, so the descriptor cannot drift from the struct
 * definition. The element type is encoded only as its size; the
 * byte-swap code picks bswap16/32/64 from the size at swap time.
 */
#define PACK_ARRAY_FIELD(Struct, field)                                      \
    {                                                                        \
        #field, PACK_ARRAY, offsetof(Struct, field),                         \
            sizeof(((Struct *)0)->field[0]),                                 \
            sizeof(((Struct *)0)->field) /                                   \
                sizeof(((Struct *)0)->field[0]),                             \
            0                                                                \
    }

/*
 * PACK_VAR_FIELD builds a descriptor for a variable-length field
 * (PACK_STRING, PACK_BYTES, or PACK_VAR_ARRAY). The data field must be
 * a pointer (char * / uint8_t * / T *), and the companion len_field
 * must be a size_t holding the byte count (STRING, BYTES) or the
 * element count (VAR_ARRAY). For PACK_VAR_ARRAY, the element size is
 * derived from sizeof(*field); for STRING and BYTES it is 1.
 *
 * The wire encoding for all three is identical: a 4-byte big-endian
 * byte count followed by the data bytes. The byte count is
 * byte-swapped like a uint32. For PACK_VAR_ARRAY each data element is
 * byte-swapped on little-endian hosts using the same path as
 * PACK_ARRAY.
 */
#define PACK_VAR_FIELD(Struct, field, ftype, len_field)                      \
    {                                                                        \
        #field, (ftype), offsetof(Struct, field),                            \
            sizeof(*(((Struct *)0)->field)), 0,                              \
            offsetof(Struct, len_field)                                      \
    }

typedef struct {
    const pack_field_t *fields;
    size_t field_count;
    size_t struct_size;
    /*
     * Optional schema version. When non-zero, the serializer emits a
     * 4-byte big-endian version prefix before the field bytes, and the
     * deserializer checks it matches. When zero (default), no prefix is
     * emitted and v0.1 wire format is preserved exactly.
     */
    uint32_t schema_version;
} pack_schema_t;

typedef enum {
    PACK_OK = 0,
    PACK_ERR_INVALID = -1,
    PACK_ERR_BUF_TOO_SMALL = -2,
    PACK_ERR_TRUNCATED = -3,
    PACK_ERR_BAD_LEN = -4,           /* variable-length byte count > UINT32_MAX,
                                      * or PACK_VAR_ARRAY byte count not a
                                      * multiple of element size */
    PACK_ERR_SCHEMA_VERSION = -5,    /* wire schema_version != schema->schema_version */
    PACK_ERR_ALLOC = -6              /* allocator returned NULL */
} pack_err_t;

/*
 * Pluggable allocator hook. If alloc is NULL, malloc is used. If free
 * is NULL, free is used. user_data is passed through to both callbacks.
 * An allocator with both alloc and free NULL is equivalent to passing
 * allocator = NULL. alloc must return memory aligned suitably for any
 * object type (matching malloc's contract).
 */
typedef struct {
    void *(*alloc)(size_t size, void *user_data);
    void  (*free)(void *ptr, size_t size, void *user_data);
    void  *user_data;
} pack_allocator_t;

/*
 * Serialize struct_ptr according to schema into out_buf (caller-owned,
 * buf_size bytes). On success writes the number of bytes emitted to
 * *out_written (out_written may be NULL). Returns PACK_OK on success,
 * PACK_ERR_INVALID on NULL arguments or a NULL data pointer with a
 * non-zero length, PACK_ERR_BAD_LEN if a variable-length field's byte
 * count exceeds UINT32_MAX or a PACK_VAR_ARRAY byte count is not a
 * multiple of the element size, PACK_ERR_BUF_TOO_SMALL if buf_size is
 * less than pack_serialized_size_var(schema, struct_ptr).
 *
 * For variable-length schemas, the caller must size out_buf using
 * pack_serialized_size_var. The caller-supplied struct's data pointers
 * are read; no allocation is performed.
 */
int pack_serialize(const pack_schema_t *schema, const void *struct_ptr,
                   void *out_buf, size_t buf_size, size_t *out_written);

/*
 * Reverse of pack_serialize. Reads wire bytes from in_buf and writes
 * them into struct_ptr's fields (after byte-swapping back to host order
 * on little-endian hosts). For variable-length fields, the caller's
 * struct must already contain valid writable pointers in the data
 * pointer fields; the deserializer writes byte_count bytes through
 * them and stores the byte count (or element count for PACK_VAR_ARRAY)
 * into each len_field. The caller is responsible for sizing the data
 * buffers; pack_deserialize_alloc is the safer path for untrusted
 * inputs. Returns PACK_OK on success, PACK_ERR_INVALID on NULL
 * arguments, PACK_ERR_TRUNCATED if in_buf is shorter than the wire
 * stream requires, PACK_ERR_BAD_LEN if a PACK_VAR_ARRAY byte count is
 * not a multiple of the element size, PACK_ERR_SCHEMA_VERSION if the
 * wire version prefix does not match schema->schema_version.
 */
int pack_deserialize(const pack_schema_t *schema, void *struct_ptr,
                     const void *in_buf, size_t buf_size);

/*
 * Sum of sizeof(field) over the schema, plus 4 if schema_version is
 * non-zero. Equal to the number of bytes emitted by pack_serialize for
 * a fixed-only schema. May be less than schema->struct_size when the
 * struct contains padding. Returns 0 for variable-length schemas (the
 * size depends on the struct instance; use pack_serialized_size_var
 * for those).
 */
size_t pack_serialized_size(const pack_schema_t *schema);

/*
 * Returns 1 if the schema contains any variable-length field
 * (PACK_STRING / PACK_BYTES / PACK_VAR_ARRAY), 0 otherwise.
 */
int pack_schema_has_variable(const pack_schema_t *schema);

/*
 * Exact serialized size for a given schema applied to a given struct
 * instance. For fixed-only schemas, equivalent to pack_serialized_size
 * (plus the 4-byte version prefix when schema_version != 0). For
 * schemas with variable-length fields, walks each variable-length
 * field's len_field and adds 4 (length prefix) + byte_count. Returns 0
 * on a NULL schema or NULL struct_ptr.
 */
size_t pack_serialized_size_var(const pack_schema_t *schema,
                                const void *struct_ptr);

/*
 * Serialize, allocating the output buffer internally via allocator
 * (NULL allocator = malloc). On success, *out_buf points to a buffer
 * of *out_size bytes owned by the caller; the caller must release it
 * with allocator->free (or free if allocator is NULL). Returns
 * PACK_OK, PACK_ERR_INVALID, PACK_ERR_BAD_LEN, or PACK_ERR_ALLOC.
 */
int pack_serialize_alloc(const pack_schema_t *schema,
                         const void *struct_ptr,
                         void **out_buf, size_t *out_size,
                         const pack_allocator_t *allocator);

/*
 * Deserialize, allocating the struct internally via allocator (NULL
 * allocator = malloc). The base struct is allocated as
 * schema->struct_size bytes; each variable-length field's data buffer
 * is allocated separately as byte_count bytes (or skipped if
 * byte_count is 0, in which case the field's data pointer is set to
 * NULL). The caller owns *out_struct and must release it with
 * pack_free_struct using the same allocator. Returns PACK_OK,
 * PACK_ERR_INVALID, PACK_ERR_TRUNCATED, PACK_ERR_BAD_LEN,
 * PACK_ERR_SCHEMA_VERSION, or PACK_ERR_ALLOC. On failure any partial
 * allocation is rolled back via pack_free_struct before returning.
 */
int pack_deserialize_alloc(const pack_schema_t *schema,
                           const void *in_buf, size_t in_size,
                           void **out_struct,
                           const pack_allocator_t *allocator);

/*
 * Release a struct previously produced by pack_deserialize_alloc.
 * Frees each variable-length field's data buffer (if non-NULL) and
 * then the struct itself, using the same allocator. Calling this on a
 * struct not produced by pack_deserialize_alloc, or with a different
 * allocator, is undefined behavior. No-op (returns PACK_OK) if
 * struct_ptr is NULL.
 */
int pack_free_struct(const pack_schema_t *schema, void *struct_ptr,
                     const pack_allocator_t *allocator);

/*
 * SIMD-accelerated byte-swap primitives. Exposed because they are useful
 * on their own; pack_serialize/deserialize call them for array fields.
 *
 * pack_bswap_uint32_array swaps `count` 32-bit elements in place using
 * the best SIMD path available on the host CPU, with a scalar BSWAP tail
 * for the trailing elements. pack_bswap_uint64_array swaps `count`
 * 64-bit elements in place the same way. pack_bswap_float_array and
 * pack_bswap_double_array do the same for IEEE-754 floats and doubles,
 * treating them as their integer counterparts. This is correct on any
 * host where float byte order matches integer byte order (every x86-64,
 * every little-endian AArch64).
 *
 * The variant actually executed is chosen at program load time by an
 * ifunc resolver (see src/pack_dispatch.c), in priority order:
 *   avx512vbmi  - VPERMB zmm, 64-byte parallel byte permute
 *   avx2        - VPSHUFB ymm, 32-byte parallel byte shuffle
 *   ssse3       - PSHUFB xmm, 16-byte parallel byte shuffle
 *   scalar      - BSWAP r32 / r64 loop, baseline x86-64
 * The library compiles without -mavx2 / -mssse3 / -mavx512* in the
 * baseline CFLAGS; each variant is enabled per-function via
 * __attribute__((target("..."))). The resulting binary runs on any
 * x86-64 CPU and uses the best available SIMD ISA at runtime.
 */
void pack_bswap_uint32_array(uint32_t *arr, size_t count);
void pack_bswap_uint64_array(uint64_t *arr, size_t count);
void pack_bswap_float_array(float *arr, size_t count);
void pack_bswap_double_array(double *arr, size_t count);

#endif /* PACK_H */
