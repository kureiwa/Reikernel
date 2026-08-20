# libpack: Design Notes (v0.3)

## Problem

Serialize C structs to a compact, portable (endianness-correct) binary
format without a separate code-generation step (unlike protobuf/`protoc`),
targeting config files, network packet headers, and save-game state per
the original notes' use cases.

## Why macro-generated field tables, not hand-written

A hand-written offsetof/sizeof table is possible but error-prone (easy to
typo an offset or forget to update it after a struct change, with no
compile-time check). The `PACK_FIELD` macro pattern computes `offsetof`
and `sizeof` directly from the struct definition at compile time, so a
mismatched struct/table pair is far less likely. Worth the minor added
macro complexity.

## Why SIMD endianness conversion is core

SIMD endianness conversion is core to this project's stated goal (a
mastery/portfolio project centered on ASM/ABI work). Shipping a serializer
that is just `memcpy` + a scalar `bswap` loop would miss the entire reason
this module exists rather than just using an existing serialization
library. `_mm256_shuffle_epi8` (VPSHUFB ymm, 32 bytes wide) handles 8
`uint32_t` elements per instruction, or 4 `uint64_t` elements per
instruction. (16 `uint16_t` per instruction is also possible; `PACK_U16`
is in the enum and exercised by `test_basic` and `test_endian`, but no
SIMD path for 16-bit element arrays is implemented yet -- they fall through
to the scalar `__builtin_bswap16` loop in `pack_swap_region`. 16-wide SIMD
bswap is an unclaimed perf opportunity.) Scalar fallback
(`__builtin_bswap32` / `64`, which compile to single `BSWAP` instructions)
is still needed for single struct members and for array lengths not
divisible by the SIMD width. SIMD loads/stores use the unaligned intrinsics
(`_mm256_loadu_si256` / `_mm256_storeu_si256`) because struct field
addresses are not guaranteed 32-byte aligned; the AVX-512 variant
additionally probes for 64-byte alignment at function entry and switches
to the aligned intrinsics when it holds (see "AVX-512 alignment fast path"
below).

Note: `__builtin_bswap16` does NOT compile to `BSWAP` on x86 -- x86
`BSWAP` has no 16-bit form, so the compiler emits `rolw $8, %ax`.
Single-instruction, correct, but not `bswap`.

## Runtime CPU dispatch (v0.3)

Earlier builds compiled `pack_simd.c` with `-mavx2 -mssse3` in the baseline
`CFLAGS`, which made the resulting binary require AVX2 at runtime: a
pre-Haswell CPU (pre-2013) would `SIGILL` on the first `VPSHUFB ymm`. The
current build replaces this with a runtime dispatch. The approach:

- **Baseline CFLAGS** carry no `-mavx2` / `-mssse3` / `-mavx512*`
  flags. The whole library compiles for baseline x86-64.
- **Per-function ISA** is enabled with `__attribute__((target("...")))`
  on each variant in `src/pack_simd.c`. The variant's body is emitted
  with the requested ISA enabled (e.g. `target("avx512vbmi")` lets the
  function use `VPERMB zmm`), but the rest of the TU is unaffected.
  The variant functions are ordinary external symbols; the target
  attribute only changes how the compiler generates that function's
  body.
- **Public symbol resolution** uses GCC's `ifunc` attribute. The
  public symbols `pack_bswap_uint32_array` and
  `pack_bswap_uint64_array` are declared in `src/pack_dispatch.c`
  with `__attribute__((ifunc("resolver")))`. The linker emits an
  `R_X86_64_IRELATIVE` relocation against the resolver name. At
  program load time the dynamic linker (or, for fully-static links,
  the libc startup code) calls the resolver once, stores the returned
  function pointer in the GOT, and every later call goes directly to
  the chosen variant. There is no per-call branch and no function-
  pointer table lookup; the call site is just a normal indirect call
  through the GOT, same as any PLT call.

CPU feature detection uses `__builtin_cpu_supports("avx2")` etc.,
which queries the `AT_HWCAP` / `AT_HWCAP2` bits the kernel recorded
for the process at `exec` time. The resolver runs before `main` and
must not touch TLS, errno, `malloc`, or any libc state that is not
yet initialized; `__builtin_cpu_supports` is safe because it reads a
static const struct populated by libc's own initialization
constructor, which runs during the loader's RELRO setup phase --
before any `IRELATIVE` resolver is invoked.

Selection order (most preferred first):

| ISA           | Instruction    | Width  | uint32 / instruction | uint64 / instruction |
|---------------|----------------|--------|----------------------|----------------------|
| avx512vbmi    | VPERMB zmm     | 64 B   | 16                   | 8                    |
| avx2          | VPSHUFB ymm    | 32 B   | 8                    | 4                    |
| ssse3         | PSHUFB xmm     | 16 B   | 4                    | 2                    |
| scalar        | BSWAP r32/r64  | 1 elmt | 1                    | 1                    |

The chain is a strict superset: every CPU that supports a higher tier
also supports every lower tier. SSSE3 has been baseline on x86-64
since Core 2 (2006), so the scalar path is in practice reserved for
pre-2006 hardware (NetBurst Pentium 4 with EM64T, early K8). The
AVX-512 VBMI variant requires Ice Lake client (2019) / Cascade Lake
server (2019) or later; older AVX-512 hardware that has AVX-512F but
not VBMI (e.g. Skylake-X with AVX-512F/BW/CD/DQ/VL) correctly falls
through to AVX2, because `VPERMB` is gated on VBMI specifically, not
just AVX-512F.

### Why `ifunc` and not a function-pointer init pattern

The alternative is a `static pack_bswap32_fn g_bswap32 = resolve();`
initialized by a `__attribute__((constructor))` function, with the
public symbol re-routed through that pointer. This works but adds a
per-call memory load (the function pointer) and an extra branch
through the constructor machinery. `ifunc` is the loader-native
mechanism for this exact pattern: the GOT slot is populated once by
the loader, every call site is a normal indirect call through the GOT,
and there is no constructor to schedule. The cost is exactly one
indirect call per invocation, same as any PLT call -- which the
compiler already does for cross-TU function calls unless link-time
optimization inlines them.

### Why per-function `target` attributes and not separate TUs

The alternative is one `.c` file per ISA (`pack_simd_avx2.c`,
`pack_simd_ssse3.c`, etc.), each compiled with the appropriate `-m`
flag, linked into the same archive. This works but multiplies the
source files and the Makefile rules. Per-function `target` attributes
keep all the variants for one element width in one file, sharing the
masks and the scalar tail loop, and let the Makefile stay simple.

The intrinsic headers (`<immintrin.h>` and its inner includes like
`<tmmintrin.h>`, `<avx2intrin.h>`, `<avx512vbmiintrin.h>`) declare
their intrinsic functions with their own `target` attributes, so the
intrinsics are callable from any function that has the matching
`target` attribute even if the baseline CFLAGS do not enable the ISA
globally. No `#pragma GCC target` wrapper is needed around the
`#include`.

### Variant set

For each element width (32, 64) there are four variants, one per ISA
tier:

- `pack_bswap_uint{32,64}_array_scalar` -- no target attribute;
  `BSWAP r32` / `BSWAP r64` loop. Always available, baseline x86-64.
- `pack_bswap_uint{32,64}_array_ssse3` -- `target("ssse3")`; `PSHUFB
  xmm`, 16-byte per instruction (4 uint32_t or 2 uint64_t) with a
  scalar tail.
- `pack_bswap_uint{32,64}_array_avx2` -- `target("avx2")`; `VPSHUFB
  ymm`, 32-byte per instruction (8 uint32_t or 4 uint64_t) with a
  scalar tail.
- `pack_bswap_uint{32,64}_array_avx512` -- `target("avx512vbmi")`;
  `VPERMB zmm`, 64-byte per instruction (16 uint32_t or 8 uint64_t)
  with a scalar tail. `VPERMB` is a true cross-lane byte permute at
  the 64-byte granularity (each output byte can be any of the 64
  source bytes, 6-bit index, 0x3F mask), unlike `VPSHUFB ymm` which
  is two independent 16-byte lane-local permutes. This is the only
  place the AVX-512 mask differs from the AVX2 mask at the
  element-size level -- the per-element byte pattern (reverse 4 or 8
  bytes) is identical, but the AVX-512 mask does not need the
  per-128-bit-lane repetition that the AVX2 / SSSE3 mask needs.

The `pack_bswap_float_array` and `pack_bswap_double_array` wrappers
are plain (non-ifunc) functions in `pack_simd.c` that call the
ifunc-dispatched `pack_bswap_uint32_array` / `pack_bswap_uint64_array`.
The runtime variant selection applies transparently: a float array
swap on an AVX-512 VBMI host uses `VPERMB zmm`, on an AVX2 host uses
`VPSHUFB ymm`, etc.

### AVX-512 alignment fast path

The AVX-512 variants (`pack_bswap_uint32_array_avx512`,
`pack_bswap_uint64_array_avx512`) check the array pointer for 64-byte
alignment at function entry with `int aligned = ((uintptr_t)p % 64 == 0);`.
When aligned, every subsequent 64-byte block consumed by `VPERMB zmm`
is also 64-byte aligned (because 64 is the access width), so the
variant uses `_mm512_load_si512` / `_mm512_store_si512` instead of
the unaligned `_mm512_loadu_si512` / `_mm512_storeu_si512`. On
Skylake-X and later, aligned 512-bit loads avoid the L1
unaligned-split penalty.

The check is runtime, not compile-time: the same library binary
handles both aligned and unaligned callers. The cost is one AND of
the low 6 bits plus one branch at function entry, taken or not-taken
once per call. The two loop bodies (aligned / unaligned) are
otherwise identical and both feed the same scalar tail, so the
observable behavior is bit-identical to the unaligned-only path that
preceded this optimization.

The check is AVX-512-only because (a) the 64-byte load width is where
the unaligned penalty is largest on current Intel hardware and (b)
the SSSE3 / AVX2 variants use 16-byte / 32-byte accesses, for which
the unaligned penalty on Haswell-and-later is small enough that
duplicating the loop body is not worth the code-size cost. The
public-API contract ("works on any alignment") is unchanged; only
the internal load/store intrinsic differs when alignment happens to
hold.

### Dispatch visibility

The chosen variant is not directly visible from C. The resolver runs
before `main`, so by the time any caller invokes
`pack_bswap_uint32_array`, the GOT slot already points at the chosen
variant. `tests/test_dispatch.c` reports which ISAs the host has and
exercises every variant the host can actually run (variants whose ISA
the host lacks are skipped; calling them would `SIGILL`). On a host
without AVX-512 VBMI the AVX-512 variant is compiled in but never
selected and never executed; its instructions sit in the binary as
dead code.

## Endianness detection

Compile-time endianness detection uses the GCC/Clang predefined macros
`__BYTE_ORDER__`, `__ORDER_LITTLE_ENDIAN__`, `__ORDER_BIG_ENDIAN__`. C23's
`<stdbit.h>` is not used because the project targets C11.

If `__BYTE_ORDER__` is not defined at all (e.g., MSVC, some embedded
compilers), the build fails with `#error "unrecognized endianness;
__BYTE_ORDER__ not defined"`. This is intentional: silently defaulting to
the no-swap (big-endian wire == native) path on a little-endian host
without the macro would produce little-endian wire bytes and break interop
with no diagnostic. A second `#error` covers the case where `__BYTE_ORDER__`
is defined to a value other than little or big.

On `__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__`, byte-swap is applied to
fields when serializing to the (big-endian) wire format and back when
deserializing. On `__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__`, no swap is
needed.

## Float byte-swap

`PACK_F32` / `PACK_F64` are byte-swapped by `memcpy`-ing the float into a
`uint32_t` / `uint64_t`, applying `__builtin_bswap32` / `__builtin_bswap64`,
and `memcpy`-ing back. This is correct on IEEE-754 hosts where float byte
order matches integer byte order (all x86-64, all little-endian AArch64).
Directly casting `float` to `uint32_t` is forbidden (strict-aliasing
violation); `union` type-punning is technically UB on read of the non-active
member; `memcpy` is the clean, portable approach.

## Padding policy

Padding bytes between struct fields and at the struct tail are NOT
serialized. The output is a tight concatenation of field bytes. The
`PACK_FIELD` macro records `offsetof(field)` and `sizeof(field)`, so the
serializer emits only the field's bytes (contiguous `sizeof(field)` bytes
starting at `offsetof(field)`) and skips the gap between the end of one
field and the start of the next. `pack_serialized_size()` returns
`sum(sizeof(field))` over the schema, which may be less than `sizeof(Struct)`.

For variable-length fields, only the 4-byte length prefix and the data
bytes are emitted; the struct's `len_field` (a `size_t`) and the data
pointer (a `T *`) are not themselves serialized -- only the data they
reference.

## Why binary-only

Text formats (JSON/TOML) require an entirely different code path (string
formatting/parsing, escaping, no fixed size) that doesn't share much with
the binary packer beyond schema description. Splitting this out keeps the
module scoped to the core, ASM-relevant packing problem; text formats can
reuse the same `pack_schema_t` later without touching the binary path.

## Variable-length fields

Three variable-length field types cover the cases that fixed-layout-only
schemas cannot: config structs with optional names, packet payloads of
variable size, save-game arrays whose length depends on play state.

- `PACK_STRING` -- UTF-8 bytes. Field is `char *`, len_field is byte count.
- `PACK_BYTES` -- raw bytes. Field is `uint8_t *`, len_field is byte count.
- `PACK_VAR_ARRAY` -- fixed-size elements. Field is `T *`, len_field is
  element count; wire byte count is `element_count * element_size`.

All three share the same wire encoding: a 4-byte big-endian byte count
followed by the data bytes. `PACK_VAR_ARRAY` data elements are
byte-swapped on little-endian hosts using the same path as `PACK_ARRAY`.

The 4-byte length prefix caps a single field at 4 GiB. The serializer
rejects a `PACK_VAR_ARRAY` whose `element_count * element_size` would
exceed `UINT32_MAX` with `PACK_ERR_BAD_LEN`. The check is structured as
`if (element_count > UINT32_MAX / element_size) return BAD_LEN;` BEFORE
the multiply, because the multiply itself can wrap on 64-bit hosts and
bypass the post-multiply `byte_count > UINT32_MAX` check. The audit probe
that motivated this guard was `len = (1ull << 62) + 1, size = 4`, which
wraps `byte_count` to 4 and then segfaults in
`pack_bswap_uint32_array` without the fix.

`pack_serialized_size()` returns 0 for any schema containing a
variable-length field, forcing the caller to use
`pack_serialized_size_var()` (which walks the struct instance) to size
the output buffer. A caller that forgets fails loudly with
`PACK_ERR_BUF_TOO_SMALL` rather than silently underallocating.

## Pluggable allocator

`pack_serialize` / `pack_deserialize` write directly into caller-supplied
buffers and perform no allocation. `pack_serialize_alloc` /
`pack_deserialize_alloc` / `pack_free_struct` cover callers who want the
library to manage buffer sizing, especially for untrusted input where the
caller cannot safely pre-size the data buffers a `PACK_STRING` /
`PACK_BYTES` / `PACK_VAR_ARRAY` will be unpacked into.

`pack_allocator_t` is a struct of `{alloc, free, user_data}`. NULL
`allocator` (or both functions NULL) falls back to libc `malloc` / `free`.
A counting/pooling/custom allocator plugs in by populating all three. The
contract permits a half-populated allocator (one function NULL, the other
not), in which case the missing half falls back to libc -- this is
permitted by the type but is a caller bug; a future revision may tighten
the contract to both-or-neither.

`pack_deserialize_alloc` allocates the base struct plus one buffer per
non-empty variable-length field. On any failure mid-walk (TRUNCATED,
BAD_LEN, ALLOC, SCHEMA_VERSION), the partial state is rolled back via
`pack_free_struct` before returning, so the caller never owns a partial
struct. `pack_free_struct` itself is idempotent on a single struct
(NUL-out each freed pointer so a double-free is a no-op) and is a no-op
on a NULL struct pointer.

## Schema versioning

`pack_schema_t.schema_version` is a `uint32_t`. When non-zero, the
serializer emits a 4-byte big-endian version prefix before the field
bytes and the deserializer compares it against the wire value, returning
`PACK_ERR_SCHEMA_VERSION` on mismatch. When zero, no prefix is emitted
and the unversioned wire format is preserved (callers that previously
serialized without a version can adopt a non-zero schema_version without
re-encoding their existing data, provided they keep the version at 0
until they actually bump).

This is an exact-match check, not a version-negotiation protocol: if a
schema bumps its version, the deserializer rejects old wire data rather
than attempting to migrate. Migration is the caller's responsibility
(invoke a schema-specific upgrade function before deserializing with the
new schema).

## pack_serialize pre-walk skip (v0.3)

`pack_serialize` no longer calls `pack_serialized_size_var` upfront to
pre-size the output buffer. The core serializer `pack_serialize_into`
already bounds-checks every field against the remaining buffer
(`buf_size - pos`), so the pre-walk is redundant work: it walks the schema
a second time just to compute the same total that `pack_serialize_into`
produces incrementally. The only pre-check that remains is room for the
4-byte version prefix (when `schema_version != 0`), because the prefix is
written before `pack_serialize_into` is called.

The API contract is unchanged. On error (`PACK_ERR_BUF_TOO_SMALL` /
`PACK_ERR_BAD_LEN`) the version prefix may already have been written; the
contract is that `*out_written` is only written on success and the caller
must not inspect `out_buf` on a non-OK return, so a partial prefix write is
safe. For schemas with no version prefix, no write occurs before
`pack_serialize_into`'s first per-field bounds check, so the
"reject-without-writing" property is preserved exactly.

Measured impact: on a 64-byte single-array struct, this trims one full
schema-walk function call (and its `pack_is_var_type` per-field
comparisons) off every serialize, bringing the serialize path within ~1 ns
of the deserialize path (which already had no pre-walk). The
ifunc-resolved SIMD path itself is unchanged.

## Small-count scalar fast path (v0.3)

`pack_swap_region` (the per-field swap entry point in `pack.c`) takes a
fast path for short runs: `uint32_t` arrays of 4 or fewer elements and
`uint64_t` arrays of 2 or fewer elements are byte-swapped with a tight
scalar `__builtin_bswap32` / `__builtin_bswap64` loop, and the
ifunc-dispatched SIMD helper is not called.

Every SIMD variant falls through to a scalar tail for counts below its
register width (16 for avx512, 8 for avx2, 4 for ssse3), so for these
small counts the SIMD call would add only the indirect-call + mask-load +
alignment-check + loop-setup overhead without executing a SIMD
instruction. The scalar loop wins for counts at or below the SSSE3 width;
the threshold is set conservatively at 4 for `uint32_t` and 2 for
`uint64_t` (16 bytes either way, the SSSE3 register width).

This is internal to `pack_swap_region`; the public
`pack_bswap_uint32_array` / `pack_bswap_uint64_array` symbols are
unaffected and remain ifunc-dispatched for direct callers (verified by
`tests/test_dispatch.c`, which exercises every variant the host can run).

## Benchmarks

Measured on a host with AVX-512 VBMI (resolver selects
`pack_bswap_uint{32,64}_array_avx512`). Numbers vary with CPU frequency
state and cache warmth; the figures below are representative single-run
observations from `bench/bench_serialize`, `bench/bench_serialize_large`,
and `bench/bench_variable` (1,000,000 iterations for the 64-byte case,
200,000 for the larger and variable cases).

| Bench                                | Direction   | Result            |
|--------------------------------------|-------------|-------------------|
| 64-byte struct (16 x `uint32_t`)     | serialize   | 15.5 ns/op        |
| 64-byte struct (16 x `uint32_t`)     | deserialize | 13.5 ns/op        |
| 4 KB struct (1024 x `uint32_t`)      | either      | ~12.7 GB/s        |
| 4 KiB `PACK_BYTES` + 256-elem `PACK_VAR_ARRAY` (5132 B wire) | deserialize (allocator path) | ~39 GB/s |

The 64-byte case is dominated by per-call schema-walk + ifunc dispatch +
mask-load overhead; the SIMD work is exactly one `VPERMB zmm`. The 4 KB
case is dominated by SIMD throughput (64 `VPERMB zmm`); per-call overhead
is negligible. The variable-length case is dominated by the 4 KiB `memcpy`
plus the 1 KiB SIMD swap on the 256-element `uint32_t` array.

## Non-goals

- Not a wire-protocol/RPC system: no schema evolution, no field-tag
  negotiation, no forward/backward compatibility across versions. The
  version prefix is a hard match.
- Not attempting nested struct support.
- Not a JSON/TOML library.

Confirmed scope: fixed-size primitives, fixed-size arrays, and
length-prefixed variable-length fields (strings, byte blobs, dynamic
arrays of fixed-size elements). No nested structs.
