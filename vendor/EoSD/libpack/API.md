# libpack: API (v0.3)

Status: shipped; tests in `tests/` and `tests/extreme/`.

## Overview

Compile-time-described struct (de)serialization into a compact, packed,
endianness-correct binary format. Field layout described via macro-generated
tables. SIMD (AVX-512 VBMI / AVX2 / SSSE3) endianness conversion for arrays
of 32- and 64-bit primitives, with the variant selected at runtime via GCC's
`ifunc` mechanism so a single library build runs on any x86-64 CPU. Binary
format only; JSON/TOML text output is out of scope.

Three variable-length field types (`PACK_STRING`, `PACK_BYTES`,
`PACK_VAR_ARRAY`) cover UTF-8 strings, raw byte blobs, and dynamic arrays of
fixed-size elements. An optional pluggable allocator (`pack_allocator_t`)
backs the serialize-into-heap / deserialize-from-heap entry points, and an
optional `schema_version` prefix lets the deserializer reject mismatched
wire data. The caller-supplied-buffer entry points (`pack_serialize`,
`pack_deserialize`) work for both fixed-only and variable-length schemas.

The library is built without `-mavx2` / `-mssse3` / `-mavx512*` in the
baseline `CFLAGS`; each SIMD variant is compiled with a per-function
`__attribute__((target("...")))` and selected by an `ifunc` resolver.
Callers link the same library and call the same functions regardless of
host CPU; the resolver transparently picks the fastest path the host can
run.

## Field description macros

```c
typedef enum {
    PACK_U8, PACK_U16, PACK_U32, PACK_U64,
    PACK_I8, PACK_I16, PACK_I32, PACK_I64,
    PACK_F32, PACK_F64,
    PACK_ARRAY,         // fixed-size array; element type encoded as size

    /*
     * v0.2 variable-length field types. Each is serialized as a 4-byte
     * big-endian byte count followed by the data bytes. For
     * PACK_VAR_ARRAY the byte count is element_count * element_size.
     */
    PACK_STRING,        // UTF-8 bytes; field is char *, len_field is bytes
    PACK_BYTES,         // raw bytes;   field is uint8_t *, len_field is bytes
    PACK_VAR_ARRAY      // fixed-size elements; field is T *, len_field is count
} pack_field_type_t;

typedef struct {
    const char *name;
    pack_field_type_t type;
    size_t offset;          // offsetof(Struct, field)
    size_t size;            // sizeof(field), or element size for arrays
                            // and PACK_VAR_ARRAY (1 for STRING/BYTES)
    size_t array_count;     // 0 for scalar fields, element count for
                            // fixed-size arrays, unused for variable-length
    size_t len_offset;      // offsetof(Struct, len_field) for variable-
                            // length fields; 0 otherwise
} pack_field_t;

// Scalar field descriptor.
#define PACK_FIELD(Struct, field, ftype) \
    { #field, (ftype), offsetof(Struct, field), \
      sizeof(((Struct*)0)->field), 0, 0 }

// Fixed-size array field descriptor.
#define PACK_ARRAY_FIELD(Struct, field) \
    { #field, PACK_ARRAY, offsetof(Struct, field), \
      sizeof(((Struct*)0)->field[0]), \
      sizeof(((Struct*)0)->field) / sizeof(((Struct*)0)->field[0]), 0 }

// Variable-length field descriptor. len_field must be a size_t holding
// the byte count (STRING, BYTES) or element count (VAR_ARRAY).
#define PACK_VAR_FIELD(Struct, field, ftype, len_field) \
    { #field, (ftype), offsetof(Struct, field), \
      sizeof(*(((Struct*)0)->field)), 0, \
      offsetof(Struct, len_field) }

typedef struct {
    const pack_field_t *fields;
    size_t field_count;
    size_t struct_size;
    /*
     * Optional schema version. When non-zero, the serializer emits a
     * 4-byte big-endian version prefix before the field bytes and the
     * deserializer rejects a non-matching wire version. When zero, no
     * prefix is emitted and the v0.1 wire format is preserved exactly.
     */
    uint32_t schema_version;
} pack_schema_t;
```

## Wire format

Big-endian, flat (no TLV). Padding bytes between fields and at the struct
tail are NOT serialized. The output is a tight concatenation of field
bytes.

If `schema_version != 0`, the first 4 bytes of the stream are the version
as a big-endian `uint32`. The deserializer compares it against
`schema->schema_version` and returns `PACK_ERR_SCHEMA_VERSION` on mismatch.

Each variable-length field (PACK_STRING / PACK_BYTES / PACK_VAR_ARRAY) is
encoded as a 4-byte big-endian byte count followed by the data bytes. For
PACK_VAR_ARRAY, each element is byte-swapped on little-endian hosts using
the same path as PACK_ARRAY. A zero byte count is encoded as `00 00 00 00`
with no following data.

A `PACK_VAR_ARRAY` whose `element_count * element_size` would exceed
`UINT32_MAX` is rejected at serialize time with `PACK_ERR_BAD_LEN`. The
check is `if (element_count > UINT32_MAX / element_size) return PACK_ERR_BAD_LEN;`
performed *before* the multiply, because the multiply itself can wrap on a
64-bit host (e.g. `len = 2^62 + 1, size = 4` wraps `byte_count` to 4) and
bypass a post-multiply `byte_count > UINT32_MAX` check. The same guard is
applied in `pack_serialized_size_var`, which returns 0 on overflow (forcing
`pack_serialize_alloc` into its `PACK_ERR_BAD_LEN` path); `pack_free_struct`
uses `SIZE_MAX` as the size hint when freeing an already-corrupted len to
avoid passing a wrapped size to a tracking allocator.

On deserialize, a `PACK_VAR_ARRAY` byte count that is not a multiple of the
element size is rejected with `PACK_ERR_BAD_LEN`.

## Error codes

```c
typedef enum {
    PACK_OK                = 0,
    PACK_ERR_INVALID       = -1,   // NULL argument, NULL data pointer with
                                   // non-zero length, struct_size == 0
    PACK_ERR_BUF_TOO_SMALL = -2,   // serialize: caller buffer < required
    PACK_ERR_TRUNCATED     = -3,   // deserialize: input shorter than wire
                                   // requires (incl. missing version prefix)
    PACK_ERR_BAD_LEN       = -4,   // variable-length byte count > UINT32_MAX,
                                   // PACK_VAR_ARRAY byte count not a multiple
                                   // of element size, or
                                   // element_count * element_size overflows
    PACK_ERR_SCHEMA_VERSION = -5,  // wire schema_version != schema->schema_version
    PACK_ERR_ALLOC         = -6,   // allocator returned NULL
} pack_err_t;
```

## Pluggable allocator

```c
typedef struct {
    void *(*alloc)(size_t size, void *user_data);
    void  (*free)(void *ptr, size_t size, void *user_data);
    void  *user_data;
} pack_allocator_t;
```

If `allocator` is NULL, or both `alloc` and `free` are NULL, libc
`malloc`/`free` are used. `alloc` must return memory aligned suitably for
any object type (matching `malloc`'s contract). Mixing a non-NULL `alloc`
with a NULL `free` (or vice versa) is permitted by the type but is a
caller bug: the corresponding libc routine is used for the missing half,
which is rarely what the caller intends. Future revisions may tighten
this to a both-or-neither contract.

## Serialize / deserialize (caller-supplied buffers)

```c
// Serializes struct_ptr according to schema into out_buf (caller-owned,
// buf_size bytes). On success writes the number of bytes emitted to
// *out_written (out_written may be NULL). For variable-length schemas
// the caller must size out_buf using pack_serialized_size_var. Returns
// PACK_OK, PACK_ERR_INVALID, PACK_ERR_BAD_LEN, or PACK_ERR_BUF_TOO_SMALL.
int pack_serialize(const pack_schema_t *schema, const void *struct_ptr,
                   void *out_buf, size_t buf_size, size_t *out_written);

// Reverse of pack_serialize. For variable-length fields the caller's
// struct must already contain valid writable pointers in the data
// pointer fields; the deserializer writes byte_count bytes through them
// and stores the byte count (or element count for PACK_VAR_ARRAY) into
// each len_field. The caller is responsible for sizing the data buffers;
// pack_deserialize_alloc is the safer path for untrusted inputs.
// Returns PACK_OK, PACK_ERR_INVALID, PACK_ERR_TRUNCATED,
// PACK_ERR_BAD_LEN, or PACK_ERR_SCHEMA_VERSION.
int pack_deserialize(const pack_schema_t *schema, void *struct_ptr,
                     const void *in_buf, size_t buf_size);

// Sum of sizeof(field) over the schema, plus 4 if schema_version is
// non-zero. Equal to the number of bytes emitted by pack_serialize for
// a fixed-only schema. Returns 0 for variable-length schemas (use
// pack_serialized_size_var for those).
size_t pack_serialized_size(const pack_schema_t *schema);

// Returns 1 if the schema contains any variable-length field, 0 otherwise.
int pack_schema_has_variable(const pack_schema_t *schema);

// Exact serialized size for a given schema applied to a given struct
// instance. For fixed-only schemas, equivalent to pack_serialized_size
// (plus the 4-byte version prefix when schema_version != 0). For
// variable-length schemas, walks each variable-length field's len_field
// and adds 4 (length prefix) + byte_count. Returns 0 on a NULL schema,
// NULL struct_ptr, or PACK_VAR_ARRAY multiply overflow.
size_t pack_serialized_size_var(const pack_schema_t *schema,
                                const void *struct_ptr);
```

## Serialize / deserialize (heap-allocated)

```c
// Serialize, allocating the output buffer internally via allocator (NULL
// allocator = malloc). On success *out_buf points to a buffer of
// *out_size bytes owned by the caller; release it with allocator->free
// (or free if allocator is NULL). Returns PACK_OK, PACK_ERR_INVALID,
// PACK_ERR_BAD_LEN, or PACK_ERR_ALLOC.
int pack_serialize_alloc(const pack_schema_t *schema,
                         const void *struct_ptr,
                         void **out_buf, size_t *out_size,
                         const pack_allocator_t *allocator);

// Deserialize, allocating the struct internally via allocator. The base
// struct is allocated as schema->struct_size bytes; each variable-length
// field's data buffer is allocated separately as byte_count bytes (or
// skipped if byte_count is 0, in which case the field's data pointer is
// NULL). The caller owns *out_struct and must release it with
// pack_free_struct using the same allocator. On failure any partial
// allocation is rolled back via pack_free_struct before returning.
// Returns PACK_OK, PACK_ERR_INVALID, PACK_ERR_TRUNCATED,
// PACK_ERR_BAD_LEN, PACK_ERR_SCHEMA_VERSION, or PACK_ERR_ALLOC.
int pack_deserialize_alloc(const pack_schema_t *schema,
                           const void *in_buf, size_t in_size,
                           void **out_struct,
                           const pack_allocator_t *allocator);

// Release a struct previously produced by pack_deserialize_alloc. Frees
// each variable-length field's data buffer (if non-NULL) and then the
// struct itself, using the same allocator. Calling this on a struct not
// produced by pack_deserialize_alloc, or with a different allocator, is
// undefined behavior. No-op (returns PACK_OK) if struct_ptr is NULL.
int pack_free_struct(const pack_schema_t *schema, void *struct_ptr,
                     const pack_allocator_t *allocator);
```

## SIMD byte-swap primitives

```c
// Swap `count` 32-bit elements in place. The actual implementation is
// chosen at program load time by an ifunc resolver in pack_dispatch.c;
// callers always invoke the same symbol regardless of host CPU.
//
// Selection order (most preferred first):
//   avx512vbmi  - VPERMB zmm, 16 elements per instruction (64 bytes)
//   avx2        - VPSHUFB ymm, 8 elements per instruction (32 bytes)
//   ssse3       - PSHUFB xmm, 4 elements per instruction (16 bytes)
//   scalar      - BSWAP r32 loop, baseline x86-64
//
// Each SIMD variant ends with a scalar BSWAP tail for the elements not
// consumed by the last SIMD instruction. The resolver runs once at load
// time (via an IRELATIVE relocation); subsequent calls go directly to
// the chosen variant with no per-call branch.
void pack_bswap_uint32_array(uint32_t *arr, size_t count);

// As above, 64-bit elements. AVX-512 VBMI: 8 per VPERMB zmm;
// AVX2: 4 per VPSHUFB ymm; SSSE3: 2 per PSHUFB xmm.
void pack_bswap_uint64_array(uint64_t *arr, size_t count);

// IEEE-754 float / double arrays, treated as their uint32 / uint64
// counterparts. Correct on any host where float byte order matches
// integer byte order (every x86-64, every little-endian AArch64).
// These wrappers go through the ifunc-dispatched uint32 / uint64
// symbols, so the same runtime variant selection applies.
void pack_bswap_float_array(float *arr, size_t count);
void pack_bswap_double_array(double *arr, size_t count);
```

### Runtime CPU dispatch (v0.3)

The library is built without `-mavx2` / `-mssse3` / `-mavx512*` in the
baseline `CFLAGS`; instead, each SIMD variant in `src/pack_simd.c` is
compiled with `__attribute__((target("...")))` so the variant's
instructions are emitted into the binary but only executed when the
resolver selects that variant.

The public symbols `pack_bswap_uint32_array` and
`pack_bswap_uint64_array` are declared with
`__attribute__((ifunc("resolver")))` in `src/pack_dispatch.c`. At
program load time the dynamic linker (or, for fully-static links, the
libc startup code) processes an `R_X86_64_IRELATIVE` relocation: it
calls the resolver once, stores the returned function pointer in the
GOT, and every later call jumps directly to the chosen variant. There
is no per-call branch, no function-pointer table lookup, and no extra
register pressure beyond a normal indirect call.

CPU feature detection uses `__builtin_cpu_supports`, which reads the
`AT_HWCAP` / `AT_HWCAP2` bits the kernel recorded for the process at
`exec` time. The resolver runs before `main` and uses no libc state
beyond the cached CPUID struct that libc's own initialization
populates.

The four-level fallback chain means the library works on any x86-64
CPU back to the original 2003 K8 / Pentium 4 EM64T. SSSE3 is present
on every x86-64 since Core 2 (2006), so the scalar path is effectively
reserved for pre-2006 hardware in practice. The AVX-512 VBMI variant
requires Ice Lake client (2019) or Cascade Lake server (2019) or
later; on older AVX-512 hardware that has AVX-512F but not VBMI
(e.g. Skylake-X), the resolver correctly falls through to AVX2 because
`VPERMB` is gated on VBMI specifically.

### Small-count scalar fast path (v0.3)

`pack_serialize` / `pack_deserialize` route each array field through the
internal `pack_swap_region`, which selects the swap width from the field's
element size. For `uint32_t` arrays of 4 or fewer elements, and for
`uint64_t` arrays of 2 or fewer elements, the scalar `__builtin_bswap32` /
`__builtin_bswap64` loop is used directly and the ifunc-dispatched SIMD
helper is not called. Every SIMD variant falls through to a scalar tail
for counts below its register width (16 for avx512, 8 for avx2, 4 for
ssse3), so for these small counts the SIMD call would add only the
indirect-call + mask-load + alignment-check + loop-setup overhead without
executing a SIMD instruction. The scalar loop wins for counts at or below
the SSSE3 width; the threshold is set conservatively at 4 for `uint32_t`
and 2 for `uint64_t` (16 bytes either way, the SSSE3 register width).

This is internal to `pack_swap_region`; the public
`pack_bswap_uint32_array` / `pack_bswap_uint64_array` symbols are
unaffected and remain ifunc-dispatched for direct callers.

### AVX-512 alignment fast path (v0.3)

The AVX-512 variants (`pack_bswap_uint32_array_avx512`,
`pack_bswap_uint64_array_avx512`) check the array pointer for 64-byte
alignment at function entry with `int aligned = ((uintptr_t)p % 64 == 0);`.
When aligned, every subsequent 64-byte block consumed by `VPERMB zmm` is
also 64-byte aligned (because 64 is the access width), so the variant uses
`_mm512_load_si512` / `_mm512_store_si512` instead of the unaligned
`_mm512_loadu_si512` / `_mm512_storeu_si512`. On Skylake-X and later,
aligned 512-bit loads avoid the L1 unaligned-split penalty.

The check is runtime, not compile-time: the same library binary handles
both aligned and unaligned callers. The cost is one AND of the low 6 bits
plus one branch at function entry, taken or not-taken once per call. The
two loop bodies (aligned / unaligned) are otherwise identical and both
feed the same scalar tail, so the observable behavior is bit-identical to
the unaligned-only path. The check is AVX-512-only because the 64-byte
load width is where the unaligned penalty is largest on current Intel
hardware; the SSSE3 / AVX2 variants use 16-byte / 32-byte accesses, for
which the unaligned penalty on Haswell-and-later is small enough that
duplicating the loop body is not worth the code-size cost. The
public-API contract ("works on any alignment") is unchanged; only the
internal load/store intrinsic differs when alignment happens to hold.

## Non-goals

- No nested struct support (struct-within-a-struct field type). Flat
  structs only; revisit if real use cases need it.
- No JSON/TOML output.
- No schema evolution / migration: `schema_version` is an exact-match
  check, not a version-negotiation protocol.
- No bit-field support: `PACK_FIELD` cannot be instantiated for a
  bit-field member because `sizeof` and `offsetof` are not applicable to
  bit-fields in C11 (constraint violation per §6.5.3.4p1 and §7.19p3).
  Any attempt to use `PACK_FIELD` on a bit-field is a hard compile error.
- No SIMD path for 16-bit element arrays: `PACK_U16` / `PACK_I16`
  fixed-size arrays via `PACK_ARRAY_FIELD` fall through to a scalar
  `__builtin_bswap16` loop. 16 `uint16_t` per VPSHUFB ymm is possible
  but unimplemented.
