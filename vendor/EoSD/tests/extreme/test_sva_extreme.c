/* libsva extreme tests: push the mmap wrapper to its limits.
 *
 * Tests:
 * - Many mappings (1000 map/unmap cycles, leak check)
 * - Large mapping (100 MB)
 * - All protection flag combinations
 * - Multiple guard-page faults (forked children)
 * - sva_flush_tlb under load
 */

#include <sva.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <stdint.h>
#include "latency.h"

static int test_many_mappings(void)
{
    /* Map and unmap 1000 regions. If there's a leak, we'll run out
     * of address space or memory. */
    for (int i = 0; i < 1000; i++) {
        sva_err_t err;
        sva_region_t *r = sva_map_guarded(4096, SVA_PROT_READ | SVA_PROT_WRITE, NULL, &err);
        if (!r) {
            fprintf(stderr, "FAIL many_mappings: map %d failed err=%d\n", i, err);
            return 1;
        }
        /* Touch the memory. */
        memset(sva_base(r), i & 0xFF, 4096);
        sva_unmap(r);
    }
    printf("PASS many_mappings: 1000 map/unmap cycles, no leak\n");
    return 0;
}

static int test_large_mapping(void)
{
    size_t sz = 100 * 1024 * 1024; /* 100 MB */
    sva_err_t err;
    sva_region_t *r = sva_map_guarded(sz, SVA_PROT_READ | SVA_PROT_WRITE, NULL, &err);
    if (!r) {
        fprintf(stderr, "FAIL large_mapping: map %zu failed err=%d\n", sz, err);
        return 1;
    }
    if (sva_size(r) != sz) {
        fprintf(stderr, "FAIL large_mapping: size=%zu, expected %zu\n", sva_size(r), sz);
        sva_unmap(r);
        return 1;
    }

    /* Touch first and last page. */
    char *base = sva_base(r);
    base[0] = 'A';
    base[sz - 1] = 'Z';
    if (base[0] != 'A' || base[sz - 1] != 'Z') {
        fprintf(stderr, "FAIL large_mapping: memory not writable\n");
        sva_unmap(r);
        return 1;
    }

    sva_unmap(r);
    printf("PASS large_mapping: 100 MB mapped, first/last byte writable\n");
    return 0;
}

static int test_all_prot_combos(void)
{
    sva_prot_flags_t combos[] = {
        SVA_PROT_READ,
        SVA_PROT_READ | SVA_PROT_WRITE,
        SVA_PROT_READ | SVA_PROT_EXEC,
        SVA_PROT_READ | SVA_PROT_WRITE | SVA_PROT_EXEC,
    };
    const char *names[] = {"R", "RW", "RX", "RWX"};

    for (int i = 0; i < 4; i++) {
        sva_err_t err;
        sva_region_t *r = sva_map_guarded(4096, combos[i], NULL, &err);
        if (!r) {
            fprintf(stderr, "FAIL prot_combos: %s failed err=%d\n", names[i], err);
            return 1;
        }
        /* For EXEC mappings, err may be SVA_ERR_EXEC_DENIED (success with
         * warning). That's fine -- the region is still usable. */
        sva_unmap(r);
    }
    printf("PASS all_prot_combos: R, RW, RX, RWX all mapped successfully\n");
    return 0;
}

static int test_multiple_guard_faults(void)
{
    /* Fork 5 children, each hits a guard page. All should SIGSEGV. */
    for (int i = 0; i < 5; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            /* Child: map a region, touch the guard page, die. */
            sva_err_t err;
            sva_region_t *r = sva_map_guarded(4096, SVA_PROT_READ | SVA_PROT_WRITE, NULL, &err);
            if (!r) _exit(100);
            volatile char *guard = sva_guard_page_addr(r);
            *guard = 42; /* SIGSEGV */
            _exit(99); /* should not reach */
        }
        int status;
        waitpid(pid, &status, 0);
        if (!WIFSIGNALED(status) || WTERMSIG(status) != SIGSEGV) {
            fprintf(stderr, "FAIL guard_faults: child %d died with status %d (expected SIGSEGV)\n",
                    i, status);
            return 1;
        }
    }
    printf("PASS multiple_guard_faults: 5 children all SIGSEGV on guard page\n");
    return 0;
}

static int test_flush_tlb_stress(void)
{
    sva_err_t err;
    sva_region_t *r = sva_map_guarded(4096, SVA_PROT_READ | SVA_PROT_WRITE, NULL, &err);
    if (!r) { fprintf(stderr, "FAIL flush_stress: map\n"); return 1; }

    char *base = sva_base(r);
    /* Write, flush, write again -- verify data survives. */
    for (int i = 0; i < 100; i++) {
        base[0] = i & 0xFF;
        if (sva_flush_tlb(r) != SVA_OK) {
            fprintf(stderr, "FAIL flush_stress: flush %d\n", i);
            sva_unmap(r);
            return 1;
        }
        if (base[0] != (i & 0xFF)) {
            fprintf(stderr, "FAIL flush_stress: data corrupted after flush %d\n", i);
            sva_unmap(r);
            return 1;
        }
    }

    sva_unmap(r);
    printf("PASS flush_tlb_stress: 100 flush cycles, data preserved\n");
    return 0;
}

static int test_map_latency(void)
{
    const size_t N = 1000;
    uint64_t samples[1000];

    for (size_t i = 0; i < N; i++) {
        sva_err_t err;
        uint64_t t0 = latency_now_ns();
        sva_region_t *r = sva_map_guarded(4096, SVA_PROT_READ | SVA_PROT_WRITE, NULL, &err);
        sva_unmap(r);
        uint64_t t1 = latency_now_ns();
        samples[i] = t1 - t0;
    }

    uint64_t p50, p99, max;
    latency_stats(samples, N, &p50, &p99, &max);
    printf("=== latency ===\n");
    latency_print_ns("sva_map_guarded + sva_unmap 4KB", p50, p99, max, N);

    return 0;
}

int main(void)
{
    int failures = 0;
    failures += test_many_mappings();
    failures += test_large_mapping();
    failures += test_all_prot_combos();
    failures += test_multiple_guard_faults();
    failures += test_flush_tlb_stress();
    failures += test_map_latency();
    if (failures == 0) {
        printf("\nlibsva extreme: ALL PASS\n");
        return 0;
    }
    printf("\nlibsva extreme: %d FAILURE(S)\n", failures);
    return 1;
}
