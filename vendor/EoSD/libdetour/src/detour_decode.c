/* libdetour v0.2: minimal x86_64 instruction length decoder.
 *
 * Scope (per DESIGN.md): correctly decode the instructions that appear in
 * real-world function prologues on Linux/x86_64. Any opcode outside this
 * set causes detour_create to fail with DETOUR_ERR_UNSUPPORTED_INSN
 * rather than guess. Extending the set later is additive.
 *
 * Supported:
 *   - Prefixes: legacy (0x66, 0x67, 0xF0 LOCK, 0xF2/0xF3 REP, segment
 *     overrides 0x2E/0x36/0x3E/0x26/0x64/0x65), REX (0x40-0x4F).
 *   - ENDBR64 (F3 0F 1E FA), 4 bytes.
 *   - NOP (0x90), 1 byte.
 *   - Multi-byte NOP (0F 1F /0), 5-9 bytes.
 *   - PUSH/POP reg (50-5F), 1 byte.
 *   - PUSH imm32 (68), PUSH imm8 (6A).
 *   - ALU r/m,r and r,r/m (00-3B): ADD/OR/ADC/SBB/AND/SUB/XOR/CMP.
 *   - ALU al,imm8 (04,0C,14,1C,24,2C,34,3C) and ALU eax,imm32 (05,...,3D).
 *   - TEST r/m,r (84/85), TEST al,imm8 (A8), TEST eax,imm32 (A9).
 *   - XCHG r/m,r (86/87).
 *   - MOV r/m,r and r,r/m (88-8B).
 *   - MOV al/eax,moffs and reverse (A0-A3); moffs is 8 bytes (4 with 0x67).
 *   - MOV r8,imm8 (B0-B7), MOV r,imm (B8-BF; imm64 with REX.W, imm16 with
 *     0x66, else imm32).
 *   - LEA r,m (8D).
 *   - JMP rel8 (EB), JMP rel32 (E9), CALL rel32 (E8).
 *   - Group 1 (80/81/82/83) r/m,imm: ADD/OR/ADC/SBB/AND/SUB/XOR/CMP.
 *   - Group 11 (C6/C7) r/m,imm: MOV (reg field must be 0).
 *   - Group 3 (F6/F7) r/m: TEST (/0,/1) takes an immediate; NOT/NEG/MUL/
 *     IMUL/DIV/IDIV (/2-/7) take no immediate.
 *   - 2-byte: 0F 1F /0 (NOP), 0F AF (IMUL r,r/m), 0F B6/B7/BE/BF
 *     (MOVZX/MOVSX r,r/m).
 *
 * RIP-relative detection: any ModR/M byte with mod=00 and rm=101 (and no
 * 0x67 address-size prefix) selects [rip+disp32]. The decoder records the
 * offset of the 4-byte displacement within the instruction and its value,
 * so the relocator in detour.c can recompute the displacement for the
 * trampoline's new address.
 */

#include "detour_internal.h"

#include <string.h>

/* Parse ModR/M, optional SIB, and displacement. `pos` is the offset of
 * the ModR/M byte within the instruction. Returns the new offset (after
 * ModR/M + SIB + displacement), or -1 on truncation. On RIP-relative
 * forms, sets out->is_rip_relative, out->disp_offset, out->disp_value.
 *
 * `available` is the number of bytes from the start of the instruction
 * (p[0]) that are readable; pos and the return value are byte offsets
 * from p[0]. */
static int parse_modrm(const uint8_t *p, size_t available, int pos,
                       int has_67, detour_insn_t *out)
{
    if (pos < 0 || (size_t)pos >= available) return -1;
    uint8_t modrm = p[pos];
    int mod = modrm >> 6;
    int rm  = modrm & 7;
    int i = pos + 1;

    if (mod == 3) {
        /* register direct; no SIB, no displacement. */
        return i;
    }

    if (mod == 0 && rm == 5) {
        /* mod=00 rm=101: RIP-relative in 64-bit mode (no 0x67), or
         * absolute 32-bit disp with 0x67. */
        if (!has_67) {
            out->is_rip_relative = 1;
            out->disp_offset = i;
            if ((size_t)i + 4 > available) return -1;
            memcpy(&out->disp_value, p + i, 4);
        }
        if ((size_t)i + 4 > available) return -1;
        return i + 4;
    }

    if (rm == 4) {
        /* SIB byte follows. */
        if ((size_t)i >= available) return -1;
        uint8_t sib = p[i];
        i++;
        int base = sib & 7;
        if (mod == 0 && base == 5) {
            /* no base register; 4-byte displacement. */
            if ((size_t)i + 4 > available) return -1;
            i += 4;
        }
    }

    if (mod == 1) {
        if ((size_t)i >= available) return -1;
        return i + 1;
    }
    if (mod == 2) {
        if ((size_t)i + 4 > available) return -1;
        return i + 4;
    }
    /* mod == 0, rm != 5, no SIB special case: no displacement. */
    return i;
}

int detour_decode_one(const uint8_t *p, size_t available, detour_insn_t *out)
{
    memset(out, 0, sizeof(*out));
    if (available == 0) return -1;

    /* ENDBR64: F3 0F 1E FA. F3 is normally a REP prefix but here is part
     * of the encoding, so check the full 4-byte sequence before the
     * generic prefix loop. */
    if (available >= 4 && p[0] == 0xF3 && p[1] == 0x0F &&
        p[2] == 0x1E && p[3] == 0xFA) {
        out->length = 4;
        return 0;
    }

    int i = 0;
    int has_66 = 0, has_67 = 0;

    /* Legacy prefixes. Multiple of the same kind are allowed (redundant). */
    while ((size_t)i < available) {
        uint8_t b = p[i];
        if (b == 0x66) { has_66 = 1; i++; continue; }
        if (b == 0x67) { has_67 = 1; i++; continue; }
        if (b == 0xF0) { i++; continue; }              /* LOCK            */
        if (b == 0xF2 || b == 0xF3) { i++; continue; } /* REPNE/REP       */
        if (b == 0x2E || b == 0x36 || b == 0x3E ||
            b == 0x26 || b == 0x64 || b == 0x65) { i++; continue; }
        break;
    }

    /* REX prefix (0x40-0x4F). Must be the last prefix before the opcode. */
    int rex = 0;
    if ((size_t)i < available && p[i] >= 0x40 && p[i] <= 0x4F) {
        rex = p[i];
        i++;
    }
    int rex_w = (rex >> 3) & 1;

    if ((size_t)i >= available) return -1;
    uint8_t op = p[i];
    i++;

    /* 2-byte opcode (0F xx). */
    if (op == 0x0F) {
        if ((size_t)i >= available) return -1;
        uint8_t op2 = p[i];
        i++;
        /* Only the forms with a ModR/M and no immediate are accepted here. */
        if (op2 != 0x1F && op2 != 0xAF &&
            op2 != 0xB6 && op2 != 0xB7 &&
            op2 != 0xBE && op2 != 0xBF) {
            return -1;
        }
        /* 0F 1F /0: multi-byte NOP requires reg field == 0. */
        if (op2 == 0x1F) {
            if ((size_t)i >= available) return -1;
            uint8_t modrm = p[i];
            if (((modrm >> 3) & 7) != 0) return -1;
        }
        int after = parse_modrm(p, available, i, has_67, out);
        if (after < 0) return -1;
        out->length = after;
        return 0;
    }

    int has_modrm = 0;
    int imm_size  = 0;
    int is_group3 = 0;   /* F6/F7: immediate presence depends on reg field */

    switch (op) {
        /* NOP. (With a preceding 0x66, "66 90" is the canonical 2-byte
         * nop; the 0x66 is already accounted for in i.) */
        case 0x90:
            out->length = i;
            return 0;

        /* PUSH/POP r64 (50-5F). 1 byte. */
        case 0x50: case 0x51: case 0x52: case 0x53:
        case 0x54: case 0x55: case 0x56: case 0x57:
        case 0x58: case 0x59: case 0x5A: case 0x5B:
        case 0x5C: case 0x5D: case 0x5E: case 0x5F:
            out->length = i;
            return 0;

        /* PUSH imm32 / imm8. */
        case 0x68: imm_size = 4; break;
        case 0x6A: imm_size = 1; break;

        /* ALU r/m,r and r,r/m (8/16/32/64-bit). */
        case 0x00: case 0x01: case 0x02: case 0x03:
        case 0x08: case 0x09: case 0x0A: case 0x0B:
        case 0x10: case 0x11: case 0x12: case 0x13:
        case 0x18: case 0x19: case 0x1A: case 0x1B:
        case 0x20: case 0x21: case 0x22: case 0x23:
        case 0x28: case 0x29: case 0x2A: case 0x2B:
        case 0x30: case 0x31: case 0x32: case 0x33:
        case 0x38: case 0x39: case 0x3A: case 0x3B:
        /* TEST r/m,r; XCHG r/m,r. */
        case 0x84: case 0x85:
        case 0x86: case 0x87:
        /* MOV r/m,r and r,r/m. */
        case 0x88: case 0x89: case 0x8A: case 0x8B:
        /* LEA r,m. */
        case 0x8D:
            has_modrm = 1;
            break;

        /* ALU al,imm8. */
        case 0x04: case 0x0C: case 0x14: case 0x1C:
        case 0x24: case 0x2C: case 0x34: case 0x3C:
            imm_size = 1;
            break;
        /* ALU eax,imm32. */
        case 0x05: case 0x0D: case 0x15: case 0x1D:
        case 0x25: case 0x2D: case 0x35: case 0x3D:
            imm_size = 4;
            break;

        /* MOV al/eax,moffs and reverse. moffs is 8 bytes (4 with 0x67). */
        case 0xA0: case 0xA1: case 0xA2: case 0xA3:
            imm_size = has_67 ? 4 : 8;
            break;

        /* TEST al,imm8 / eax,imm32. */
        case 0xA8: imm_size = 1; break;
        case 0xA9: imm_size = 4; break;

        /* MOV r8,imm8. */
        case 0xB0: case 0xB1: case 0xB2: case 0xB3:
        case 0xB4: case 0xB5: case 0xB6: case 0xB7:
            imm_size = 1;
            break;

        /* MOV r,imm: imm64 with REX.W, imm16 with 0x66, else imm32. */
        case 0xB8: case 0xB9: case 0xBA: case 0xBB:
        case 0xBC: case 0xBD: case 0xBE: case 0xBF:
            imm_size = rex_w ? 8 : (has_66 ? 2 : 4);
            break;

        /* CALL rel32, JMP rel32, JMP rel8. */
        case 0xE8: imm_size = 4; break;
        case 0xE9: imm_size = 4; break;
        case 0xEB: imm_size = 1; break;

        /* Group 1: r/m,imm. 80/82 = imm8, 81 = imm32 (imm16 with 0x66),
         * 83 = imm8 sign-extended. */
        case 0x80: case 0x82: imm_size = 1; has_modrm = 1; break;
        case 0x81:            imm_size = has_66 ? 2 : 4; has_modrm = 1; break;
        case 0x83:            imm_size = 1; has_modrm = 1; break;

        /* Group 11: r/m,imm. C6 = imm8, C7 = imm32 (imm16 with 0x66).
         * Reg field must be 0 (MOV). */
        case 0xC6: imm_size = 1; has_modrm = 1; break;
        case 0xC7: imm_size = has_66 ? 2 : 4; has_modrm = 1; break;

        /* Group 3: r/m. /0 and /1 (TEST) take an immediate; /2-/7 do not. */
        case 0xF6: is_group3 = 1; has_modrm = 1; break;
        case 0xF7: is_group3 = 1; has_modrm = 1; break;

        default:
            return -1;
    }

    if (has_modrm) {
        int after = parse_modrm(p, available, i, has_67, out);
        if (after < 0) return -1;

        /* Group 11 (C6/C7): reg field must be 0. */
        if (op == 0xC6 || op == 0xC7) {
            uint8_t modrm = p[i];
            if (((modrm >> 3) & 7) != 0) return -1;
        }

        /* Group 3 (F6/F7): /0,/1 (TEST) take an immediate. */
        if (is_group3) {
            uint8_t modrm = p[i];
            int reg = (modrm >> 3) & 7;
            if (reg == 0 || reg == 1) {
                imm_size = (op == 0xF6) ? 1 : (has_66 ? 2 : 4);
            } else {
                imm_size = 0;
            }
        }

        i = after;
    }

    if (imm_size > 0) {
        if ((size_t)i + (size_t)imm_size > available) return -1;
        i += imm_size;
    }

    out->length = i;
    return 0;
}

ssize_t detour_decode_prologue(const uint8_t *p, size_t available,
                               size_t min_bytes, detour_insn_t *insns,
                               int max_insns, int *out_count)
{
    int count = 0;
    size_t total = 0;

    while (total < min_bytes) {
        if (count >= max_insns) return -1;
        if (total >= available) return 0;  /* input exhausted */

        detour_insn_t *insn = &insns[count];
        if (detour_decode_one(p + total, available - total, insn) != 0)
            return -1;
        if (insn->length <= 0) return -1;

        total += (size_t)insn->length;
        count++;
    }

    *out_count = count;
    return (ssize_t)total;
}
