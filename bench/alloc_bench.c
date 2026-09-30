/* Allocator benchmark: the HA++ allocators (happ/lib/mem.ha) against glibc malloc
 * and Matthew Conte's C TLSF. Run it through bench/alloc_bench.py, which builds
 * everything with the same compiler and CPU flags.
 *
 * Every allocator replays exactly the same sequence of requests (fixed seed).
 * Two numbers per test:
 *   ns/op   - whole run divided by the number of operations (no timer inside)
 *   p50/p99/p99.9/max - per-operation latency from the CPU's cycle counter,
 *             timer cost subtracted. For a frame loop the tail matters most:
 *             one slow call is a dropped frame. The latency run is repeated
 *             5 times and each column keeps its lowest value, so a single
 *             interrupt from the operating system doesn't decide the result.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__x86_64__)
#include <x86intrin.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif

#include "tlsf.h"       /* Conte TLSF */

/* HA++ side: tests/alloc_test.ha */
void *t_tlsf_create(void *mem, uint64_t bytes);
void *t_tlsf_malloc(void *t, uint64_t size);
void t_tlsf_free(void *t, void *p);
void *t_tlsf_realloc(void *t, void *p, uint64_t size);
uint64_t t_sizeof_arena(void);
void t_arena_init(void *a, void *mem, uint64_t size);
void *t_arena_alloc(void *a, uint64_t size, uint64_t align);
void t_arena_reset_to(void *a, uint64_t mark);
uint64_t t_sizeof_pool(void);
uint64_t t_pool_init(void *p, void *mem, uint64_t bytes, uint64_t block, uint64_t align);
void *t_pool_alloc(void *p);
void t_pool_free(void *p, void *ptr);

/* ------------------------------------------------------------------ timing */
static inline uint64_t ticks(void) {
#if defined(__x86_64__)
    unsigned aux;
    return __rdtscp(&aux);
#elif defined(__aarch64__)
    uint64_t v;
    __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static double ns_per_tick;
static uint64_t tick_overhead;

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static void calibrate(void) {
    double t0 = now_ns();
    uint64_t c0 = ticks();
    while (now_ns() - t0 < 2e8) {}
    ns_per_tick = (now_ns() - t0) / (double)(ticks() - c0);
    enum { N = 100001 };
    static uint64_t d[N];
    for (int i = 0; i < N; i++) {
        uint64_t a = ticks();
        d[i] = ticks() - a;
    }
    qsort(d, N, sizeof d[0], cmp_u64);
    tick_overhead = d[N / 2];
}

/* ------------------------------------------------------------------ allocators */
enum { GLIBC, CONTE, HAPP_TLSF, HAPP_ARENA, HAPP_POOL };
static const char *NAMES[] = {"glibc malloc", "C TLSF (Conte)", "HA++ TLSF", "HA++ Arena", "HA++ Pool"};

static tlsf_t conte;
static void *happ_tlsf;
static void *arena, *pool;
static char *heap_mem;
static const size_t HEAP = (size_t)512 << 20;

static inline void *A_malloc(int k, size_t n) {
    switch (k) {
    case GLIBC: return malloc(n);
    case CONTE: return tlsf_malloc(conte, n);
    case HAPP_TLSF: return t_tlsf_malloc(happ_tlsf, n);
    case HAPP_ARENA: return t_arena_alloc(arena, n, 16);
    default: return t_pool_alloc(pool);
    }
}

static inline void A_free(int k, void *p) {
    switch (k) {
    case GLIBC: free(p); break;
    case CONTE: tlsf_free(conte, p); break;
    case HAPP_TLSF: t_tlsf_free(happ_tlsf, p); break;
    case HAPP_POOL: t_pool_free(pool, p); break;
    default: break;
    }
}

static inline void *A_realloc(int k, void *p, size_t n) {
    switch (k) {
    case GLIBC: return realloc(p, n);
    case CONTE: return tlsf_realloc(conte, p, n);
    default: return t_tlsf_realloc(happ_tlsf, p, n);
    }
}

static void setup(int k) {
    /* the managed memory is touched once up front, as a real-time program would */
    if (k != GLIBC) memset(heap_mem, 0, HEAP);
    if (k == CONTE) conte = tlsf_create_with_pool(heap_mem, HEAP);
    if (k == HAPP_TLSF) happ_tlsf = t_tlsf_create(heap_mem, HEAP);
    if (k == HAPP_ARENA) t_arena_init(arena, heap_mem, HEAP);
    if (k == HAPP_POOL) t_pool_init(pool, heap_mem, HEAP, 32, 16);
    if ((k == CONTE && !conte) || (k == HAPP_TLSF && !happ_tlsf)) {
        fprintf(stderr, "allocator setup failed\n");
        exit(1);
    }
}

/* ------------------------------------------------------------------ workloads */
static uint64_t rng_state;
static inline uint64_t rnd(void) {         /* xorshift64*: the same sequence for every allocator */
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 2685821657736338717ull;
}

static inline size_t log_uniform(size_t lo, size_t hi) {  /* many small sizes, a few large ones */
    double u = (double)(rnd() >> 11) / 9007199254740992.0;
    double v = lo * __builtin_pow((double)hi / lo, u);
    return (size_t)v;
}

static inline void touch(char *p, size_t n) {
    p[0] = 1;
    p[n - 1] = 2;
}

typedef struct { uint64_t ops; uint64_t *lat; uint64_t nlat; int failed; } Result;

enum { SLOTS = 8192 };
static void *slot[SLOTS];
static size_t slot_n[SLOTS];

/* Mixed: a general heap. Random slots are allocated, resized or freed; sizes 16 B .. 64 KB. */
static void run_mixed(int k, Result *r, int timed_ops, uint64_t nops) {
    memset(slot, 0, sizeof slot);
    rng_state = 0x9E3779B97F4A7C15ull;
    for (uint64_t op = 0; op < nops; op++) {
        size_t i = rnd() % SLOTS;
        uint64_t t0 = timed_ops ? ticks() : 0;
        if (!slot[i]) {
            size_t n = log_uniform(16, 65536);
            slot[i] = A_malloc(k, n);
            slot_n[i] = n;
            if (!slot[i]) { r->failed = 1; return; }
        } else if (rnd() % 8 == 0) {
            size_t n = log_uniform(16, 65536);
            void *q = A_realloc(k, slot[i], n);
            if (!q) { r->failed = 1; return; }
            slot[i] = q;
            slot_n[i] = n;
        } else {
            A_free(k, slot[i]);
            slot[i] = NULL;
        }
        if (timed_ops) r->lat[r->nlat++] = ticks() - t0;
        if (slot[i]) touch(slot[i], slot_n[i]);
    }
    for (int i = 0; i < SLOTS; i++) if (slot[i]) A_free(k, slot[i]);
    r->ops = nops;
}

/* Frame: what a renderer does every frame. 64 temporary buffers (64 B .. 256 KB),
 * then all of them are released at the end of the frame. */
static void run_frame(int k, Result *r, int timed_ops, uint64_t nframes) {
    rng_state = 0xD1B54A32D192ED03ull;
    void *p[64];
    r->ops = 0;
    for (uint64_t f = 0; f < nframes; f++) {
        for (int j = 0; j < 64; j++) {
            size_t n = log_uniform(64, 262144);
            uint64_t t0 = timed_ops ? ticks() : 0;
            p[j] = A_malloc(k, n);
            if (timed_ops) r->lat[r->nlat++] = ticks() - t0;
            if (!p[j]) { r->failed = 1; return; }
            touch(p[j], n);
        }
        uint64_t t0 = timed_ops ? ticks() : 0;
        if (k == HAPP_ARENA) {
            t_arena_reset_to(arena, 0);
        } else {
            for (int j = 63; j >= 0; j--) A_free(k, p[j]);
        }
        if (timed_ops) r->lat[r->nlat++] = (ticks() - t0) / 64;   /* per buffer released */
        r->ops += 65;
    }
}

/* Nodes: many 32-byte objects, allocated and freed in random order (up to 64k alive). */
enum { NODES = 65536 };
static void *node[NODES];
static void run_nodes(int k, Result *r, int timed_ops, uint64_t nops) {
    memset(node, 0, sizeof node);
    rng_state = 0x2545F4914F6CDD1Dull;
    for (uint64_t op = 0; op < nops; op++) {
        size_t i = rnd() % NODES;
        uint64_t t0 = timed_ops ? ticks() : 0;
        if (!node[i]) {
            node[i] = A_malloc(k, 32);
            if (!node[i]) { r->failed = 1; return; }
        } else {
            A_free(k, node[i]);
            node[i] = NULL;
        }
        if (timed_ops) r->lat[r->nlat++] = ticks() - t0;
        if (node[i]) touch(node[i], 32);
    }
    for (int i = 0; i < NODES; i++) if (node[i]) A_free(k, node[i]);
    r->ops = nops;
}

/* ------------------------------------------------------------------ driver */
static void bench(const char *name, int k, uint64_t n,
                  void (*fn)(int, Result *, int, uint64_t), uint64_t max_lat) {
    Result r = {0};
    r.lat = malloc(max_lat * sizeof(uint64_t));
    setup(k);
    fn(k, &r, 0, n);                          /* warm up: caches, glibc's arenas */
    if (r.failed) { printf("%s,%s,failed\n", name, NAMES[k]); free(r.lat); return; }
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {       /* best of 3 for the throughput number */
        setup(k);
        double t0 = now_ns();
        fn(k, &r, 0, n);
        double dt = now_ns() - t0;
        if (dt < best) best = dt;
    }
    double pct[4] = {1e30, 1e30, 1e30, 1e30};
    for (int rep = 0; rep < 5; rep++) {
        setup(k);
        r.nlat = 0;
        fn(k, &r, 1, n);
        for (uint64_t i = 0; i < r.nlat; i++) r.lat[i] = r.lat[i] > tick_overhead ? r.lat[i] - tick_overhead : 0;
        qsort(r.lat, r.nlat, sizeof(uint64_t), cmp_u64);
        const double q[4] = {0.5, 0.99, 0.999, 1.0};
        for (int j = 0; j < 4; j++) {
            double v = r.lat[(uint64_t)(q[j] * (r.nlat - 1))] * ns_per_tick;
            if (v < pct[j]) pct[j] = v;
        }
    }
    printf("%s,%s,%.2f,%.1f,%.1f,%.1f,%.1f\n", name, NAMES[k], best / r.ops, pct[0], pct[1], pct[2], pct[3]);
    fflush(stdout);
    free(r.lat);
}

int main(int argc, char **argv) {
    double scale = argc > 1 ? atof(argv[1]) : 1.0;
#if defined(__linux__)
    cpu_set_t one;                             /* stay on one core: no migrations in the middle of a run */
    CPU_ZERO(&one);
    CPU_SET(0, &one);
    sched_setaffinity(0, sizeof one, &one);
#endif
    heap_mem = aligned_alloc(64, HEAP);
    arena = aligned_alloc(64, 4096);
    pool = aligned_alloc(64, 4096);
    calibrate();
    printf("# ns_per_tick=%.4f timer_overhead_ticks=%llu\n", ns_per_tick, (unsigned long long)tick_overhead);
    printf("test,allocator,ns_per_op,p50_ns,p99_ns,p999_ns,max_ns\n");
    uint64_t n_mixed = (uint64_t)(2000000 * scale), n_frames = (uint64_t)(20000 * scale),
             n_nodes = (uint64_t)(4000000 * scale);
    int mixed[] = {GLIBC, CONTE, HAPP_TLSF};
    for (int i = 0; i < 3; i++) bench("mixed", mixed[i], n_mixed, run_mixed, n_mixed);
    int frame[] = {GLIBC, CONTE, HAPP_TLSF, HAPP_ARENA};
    for (int i = 0; i < 4; i++) bench("frame", frame[i], n_frames, run_frame, n_frames * 65);
    int nodes[] = {GLIBC, CONTE, HAPP_TLSF, HAPP_POOL};
    for (int i = 0; i < 4; i++) bench("nodes", nodes[i], n_nodes, run_nodes, n_nodes);
    return 0;
}
