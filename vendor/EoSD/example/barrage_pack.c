/* example: barrage + pack -- serialize structs allocated from an arena.
 *
 * Uses libbarrage for scratch allocation and libpack for serialization.
 * Demonstrates the pattern of allocating a temporary buffer from a
 * per-thread arena, serializing a struct into it, and resetting the
 * arena when done -- no per-allocation free needed.
 */

#include <barrage.h>
#include <pack.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    uint32_t id;
    float    x, y, z;
    uint32_t flags[4];
} entity_t;

static const pack_field_t entity_fields[] = {
    PACK_FIELD(entity_t, id,    PACK_U32),
    PACK_FIELD(entity_t, x,     PACK_F32),
    PACK_FIELD(entity_t, y,     PACK_F32),
    PACK_FIELD(entity_t, z,     PACK_F32),
    PACK_ARRAY_FIELD(entity_t, flags),
};

static const pack_schema_t entity_schema = {
    .fields = entity_fields,
    .field_count = 5,
    .struct_size = sizeof(entity_t),
};

int main(void)
{
    /* Create a 1 MB arena for scratch buffers. */
    barrage_arena_t *arena = barrage_create(1 << 20, NULL);
    if (!arena) {
        fprintf(stderr, "barrage_create failed\n");
        return 1;
    }

    /* Source struct. */
    entity_t ent = {
        .id = 42,
        .x = 1.5f, .y = -2.25f, .z = 0.0f,
        .flags = { 0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0 },
    };

    /* Allocate a scratch buffer from the arena. The serialized size is
     * sum(sizeof(field)) = 4 + 4 + 4 + 4 + 16 = 32 bytes (padding
     * between x/y/z is excluded by the pack macro). */
    size_t wire_size = pack_serialized_size(&entity_schema);
    printf("struct size: %zu bytes, serialized size: %zu bytes\n",
           sizeof(entity_t), wire_size);

    barrage_err_t err;
    void *buf = barrage_alloc(arena, wire_size, 16, &err);
    if (!buf) {
        fprintf(stderr, "barrage_alloc failed: %d\n", err);
        return 1;
    }

    /* Serialize. */
    size_t written;
    if (pack_serialize(&entity_schema, &ent, buf, wire_size, &written) != PACK_OK) {
        fprintf(stderr, "pack_serialize failed\n");
        return 1;
    }
    printf("serialized %zu bytes\n", written);

    /* Deserialize into a fresh struct. */
    entity_t ent2;
    memset(&ent2, 0, sizeof(ent2));
    if (pack_deserialize(&entity_schema, &ent2, buf, written) != PACK_OK) {
        fprintf(stderr, "pack_deserialize failed\n");
        return 1;
    }

    /* Verify round-trip. */
    if (ent2.id != ent.id || ent2.x != ent.x || ent2.y != ent.y || ent2.z != ent.z) {
        fprintf(stderr, "round-trip mismatch!\n");
        return 1;
    }
    if (memcmp(ent2.flags, ent.flags, sizeof(ent.flags)) != 0) {
        fprintf(stderr, "flags round-trip mismatch!\n");
        return 1;
    }

    printf("round-trip OK: id=%u pos=(%.2f, %.2f, %.2f)\n",
           ent2.id, ent2.x, ent2.y, ent2.z);

    /* Reset the arena -- all allocations invalidated at once. */
    barrage_reset(arena);
    printf("arena reset. used=%zu capacity=%zu\n",
           barrage_used(arena), barrage_capacity(arena));

    barrage_destroy(arena);
    return 0;
}
