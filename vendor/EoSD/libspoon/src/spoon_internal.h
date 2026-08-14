#ifndef SPOON_INTERNAL_H
#define SPOON_INTERNAL_H

#include <stdint.h>
#include "spoon.h"

/* Context save area offsets. MUST match spoon_x86_64.asm.
 * The struct is laid out so these fields come first, 8-byte aligned. */
#define SPOON_OFF_SP    0
#define SPOON_OFF_RBX   8
#define SPOON_OFF_RBP   16
#define SPOON_OFF_R12   24
#define SPOON_OFF_R13   32
#define SPOON_OFF_R14   40
#define SPOON_OFF_R15   48
#define SPOON_OFF_RIP   56
#define SPOON_OFF_MXCSR 64
#define SPOON_OFF_X87CW 68

/* Default MXCSR and x87 control word values (per System V ABI). */
#define SPOON_DEFAULT_MXCSR  0x1F80u
#define SPOON_DEFAULT_X87CW  0x037Fu

struct spoon_co {
    /* Context save area -- offsets match spoon_x86_64.asm. */
    void    *sp;        /*   0 */
    uint64_t rbx;       /*   8 */
    uint64_t rbp;       /*  16 */
    uint64_t r12;       /*  24 */
    uint64_t r13;       /*  32 */
    uint64_t r14;       /*  40 */
    uint64_t r15;       /*  48 */
    uint64_t rip;       /*  56 */
    uint32_t mxcsr;     /*  64 */
    uint16_t x87_cw;    /*  68 */
    uint16_t _pad0;     /*  70 */

    /* Metadata. */
    spoon_status_t status;
    spoon_fn       fn;
    void          *arg;
    spoon_co_t    *caller;
    void          *stack;
    size_t         stack_size;
    void         (*free_hook)(void *, size_t, void *);
    void          *free_hook_user_data;
    spoon_pool_t  *pool;
};

struct spoon_pool {
    spoon_co_t      **coroutines;
    size_t            count;
    size_t            capacity;
    size_t            default_stack_size;
    spoon_allocator_t allocator;
};

/* The NASM context switch. Saves `from` context, loads `to` context. */
void spoon_switch(spoon_co_t *from, spoon_co_t *to);

/* NASM trampoline. Entered on first switch into a new coroutine.
 * Expects rbx = pointer to spoon_co_t. */
void spoon_trampoline(void);

/* C entry called by the trampoline. Runs fn, marks DONE, yields. */
void spoon_entry(spoon_co_t *co);

#endif /* SPOON_INTERNAL_H */
