"""Randomized tests of the HA++ allocators (happ/lib/mem.ha): TLSF, Arena, Pool.

    python3 -m unittest tests/test_alloc.py -v

Every operation is checked against a model kept in Python:
  - results are aligned, inside the managed memory and never overlap a live block;
  - the bytes of every live block survive all other operations (the allocator's
    own bookkeeping never writes into user data);
  - realloc keeps the old contents;
  - a failed allocation happens only when no free block is big enough;
  - after every operation, tlsf_check() walks all blocks and lists and finds
    every invariant intact (links, flags, merged neighbours, bitmaps, byte counts).

HAPP_ALLOC_LIB=/path/liballoc_test.so runs the tests against a library built
elsewhere (tests/mutate_alloc.py uses this to check that planted bugs are caught).
"""
import ctypes
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from happ.driver import build  # noqa: E402
from happ.targets import host_target  # noqa: E402

P, U64 = ctypes.c_void_p, ctypes.c_uint64
_LIB = {}


def load_lib():
    """The allocator test library (tests/alloc_test.ha), built once for this computer."""
    if "lib" in _LIB:
        return _LIB["lib"]
    path = os.environ.get("HAPP_ALLOC_LIB")
    if not path:
        out = tempfile.mkdtemp(prefix="happ-alloc-")
        _LIB["dir"] = out
        target = host_target()
        path = build(os.path.join(ROOT, "tests", "alloc_test.ha"), [target], out, quiet=True,
                     bridges=False)[target]
    lib = ctypes.CDLL(path)
    sig = {
        "t_sizeof_tlsf": ([], U64), "t_tlsf_init": ([P, P, U64], ctypes.c_bool),
        "t_tlsf_add_pool": ([P, P, U64], ctypes.c_bool), "t_tlsf_create": ([P, U64], P),
        "t_tlsf_malloc": ([P, U64], P), "t_tlsf_free": ([P, P], None), "t_tlsf_realloc": ([P, P, U64], P),
        "t_tlsf_block_size": ([P], U64), "t_tlsf_largest_free": ([P], U64), "t_tlsf_used": ([P], U64),
        "t_tlsf_check": ([P], ctypes.c_int32),
        "t_sizeof_arena": ([], U64), "t_arena_init": ([P, P, U64], None), "t_arena_alloc": ([P, U64, U64], P),
        "t_arena_mark": ([P], U64), "t_arena_reset_to": ([P, U64], None), "t_arena_peak": ([P], U64),
        "t_sizeof_pool": ([], U64), "t_pool_init": ([P, P, U64, U64, U64], U64), "t_pool_alloc": ([P], P),
        "t_pool_free": ([P, P], None), "t_pool_owns": ([P, P], ctypes.c_bool), "t_pool_used": ([P], U64),
    }
    for name, (args, ret) in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, ret
    _LIB["lib"] = lib
    return lib


def tearDownModule():
    if "dir" in _LIB:
        shutil.rmtree(_LIB["dir"], ignore_errors=True)


# ---------------------------------------------------------------- TLSF size classes, as in mem.ha
MAX_BLOCK = (1 << 32) - 16


def adjust(size):
    return 0 if size > MAX_BLOCK else max((size + 15) & ~15, 16)


def mapping(size):
    if size < 512:
        return 0, size >> 4
    top = size.bit_length() - 1
    return top - 8, (size >> (top - 5)) ^ 32


def search_class(need):
    s = need
    if s >= 512:
        s += (1 << (s.bit_length() - 1 - 5)) - 1
    return mapping(s)


def random_size(rng, big):
    r = rng.random()
    if r < 0.02:
        return 0
    if r < 0.70:
        return int(rng.integers(1, 257))
    if r < 0.95:
        return int(rng.integers(257, 8193))
    return int(rng.integers(8193, big))


class Heap:
    """Memory handed to an allocator, with an ownership map of the live blocks."""

    def __init__(self, nbytes, misalign):
        self.raw = np.zeros(nbytes + 64, np.uint8)
        start = (-self.raw.ctypes.data) % 16 + misalign        # deliberately start off a 16-byte boundary
        self.mem = self.raw[start:start + nbytes]
        self.lo = self.mem.ctypes.data
        self.hi = self.lo + nbytes
        self.owner = np.zeros(nbytes, np.int32)                # 0 = not in a live block
        self.pattern = np.zeros(1, np.uint8)
        self.pattern_of = {}

    def span(self, p, n):
        return slice(p - self.lo, p - self.lo + n)

    def claim(self, tc, ident, p, n, what):
        tc.assertTrue(self.lo <= p and p + n <= self.hi, f"{what}: block outside the managed memory")
        s = self.span(p, n)
        tc.assertFalse(self.owner[s].any(), f"{what}: block overlaps a live block")
        self.owner[s] = ident
        value = (ident * 37 + 11) & 255
        self.pattern_of[ident] = value
        self.mem[s] = value

    def release(self, p, n):
        self.owner[self.span(p, n)] = 0

    def verify_all(self, tc, what):
        live = self.owner != 0
        if not live.any():
            return
        ids = self.owner[live]
        lut = np.zeros(int(ids.max()) + 1, np.uint8)
        for ident, value in self.pattern_of.items():
            if ident < len(lut):
                lut[ident] = value
        bad = np.flatnonzero(self.mem[live] != lut[ids])
        tc.assertEqual(len(bad), 0, f"{what}: the contents of a live block changed")


class Tlsf(unittest.TestCase):
    def setUp(self):
        self.lib = load_lib()
        self.ctrl = np.zeros(self.lib.t_sizeof_tlsf() + 16, np.uint8)
        self.t = self.ctrl.ctypes.data + (-self.ctrl.ctypes.data) % 16

    def check(self, what):
        code = self.lib.t_tlsf_check(self.t)
        self.assertEqual(code, 0, f"{what}: tlsf_check failed (check {code})")

    def expect_oom(self, need, what):
        """A null result is only allowed when no list that tlsf_malloc may search holds a block."""
        if need == 0:
            return
        want = search_class(need)
        largest = self.lib.t_tlsf_largest_free(self.t)
        if want[0] < 24 and largest:
            self.assertLess(mapping(largest), want, f"{what}: returned null with a big enough free block")

    def run_random(self, seed, steps, heap_bytes, misalign=0, extra_pool=0, check_every=1):
        lib, t = self.lib, self.t
        rng = np.random.default_rng(seed)
        heap = Heap(heap_bytes, misalign)
        self.assertTrue(lib.t_tlsf_init(t, heap.lo, heap_bytes))
        heaps = [heap]
        if extra_pool:
            h2 = Heap(extra_pool, (misalign + 5) % 16)
            self.assertTrue(lib.t_tlsf_add_pool(t, h2.lo, extra_pool))
            heaps.append(h2)
        self.check("init")
        full = lib.t_tlsf_largest_free(t)
        live = {}                         # ident -> (ptr, requested, block size, heap)
        ident = 0

        def heap_of(p):
            for h in heaps:
                if h.lo <= p < h.hi:
                    return h
            self.fail(f"pointer {p:#x} is in no pool")

        big = max(9000, heap_bytes // 4)
        for step in range(steps):
            what = f"seed {seed} step {step}"
            r = rng.random()
            if not live or r < 0.5:                                   # malloc
                size = random_size(rng, big)
                p = lib.t_tlsf_malloc(t, size)
                if not p:
                    self.expect_oom(adjust(size), what)
                else:
                    self.assertEqual(p % 16, 0, f"{what}: unaligned result")
                    bs = lib.t_tlsf_block_size(p)
                    self.assertGreaterEqual(bs, max(size, 1), f"{what}: block smaller than asked")
                    h = heap_of(p)
                    ident += 1
                    h.claim(self, ident, p, bs, what)
                    live[ident] = (p, size, bs, h)
            elif r < 0.8:                                             # free
                k = list(live)[int(rng.integers(len(live)))]
                p, size, bs, h = live.pop(k)
                h.release(p, bs)
                lib.t_tlsf_free(t, p)
            else:                                                     # realloc
                k = list(live)[int(rng.integers(len(live)))]
                p, size, bs, h = live[k]
                new = random_size(rng, big) or 1
                keep = h.mem[h.span(p, min(size, new))].copy()
                q = lib.t_tlsf_realloc(t, p, new)
                if not q:
                    self.expect_oom(adjust(new), what)
                    self.assertTrue((h.mem[h.span(p, min(size, new))] == keep).all(),
                                    f"{what}: a failed realloc changed the block")
                else:
                    del live[k]
                    h.release(p, bs)
                    h2 = heap_of(q)
                    self.assertEqual(q % 16, 0)
                    self.assertTrue((h2.mem[h2.span(q, len(keep))] == keep).all(),
                                    f"{what}: realloc lost the old contents")
                    nbs = lib.t_tlsf_block_size(q)
                    self.assertGreaterEqual(nbs, new)
                    ident += 1
                    h2.claim(self, ident, q, nbs, what)
                    live[ident] = (q, new, nbs, h2)
            self.assertEqual(lib.t_tlsf_used(t), sum(v[2] for v in live.values()), f"{what}: used bytes")
            if step % check_every == 0:
                self.check(what)
            if step % 64 == 0:
                for h in heaps:
                    h.verify_all(self, what)
        for h in heaps:
            h.verify_all(self, "end")
        for k in list(live):
            p, size, bs, h = live.pop(k)
            lib.t_tlsf_free(t, p)
        self.check("all freed")
        self.assertEqual(lib.t_tlsf_used(t), 0)
        self.assertEqual(lib.t_tlsf_largest_free(t), full, "free memory didn't merge back into one block")
        return ident

    def test_random_small_heap(self):
        # a small heap runs out of memory often: exercises the failure paths
        for seed in range(6):
            with self.subTest(seed=seed):
                self.run_random(seed, 1500, 48 * 1024, misalign=seed)

    def test_random_large_heap(self):
        self.assertGreater(self.run_random(100, 6000, 1 << 21, misalign=3, check_every=7), 3000)

    def test_two_pools(self):
        self.run_random(7, 3000, 64 * 1024, misalign=9, extra_pool=40 * 1024)

    def test_exact_behaviour(self):
        lib, t = self.lib, self.t
        heap = Heap(1 << 16, 0)
        lib.t_tlsf_init(t, heap.lo, 1 << 16)
        full = lib.t_tlsf_largest_free(t)
        self.assertEqual(full, (1 << 16) - 32)
        a = lib.t_tlsf_malloc(t, 1)
        self.assertEqual(lib.t_tlsf_block_size(a), 16)          # minimum block
        self.assertEqual(a, heap.lo + 16)                       # first block, right after its header
        z = lib.t_tlsf_malloc(t, 0)
        self.assertTrue(z)                                      # malloc(0) gives a minimum block
        self.assertEqual(lib.t_tlsf_block_size(lib.t_tlsf_malloc(t, 100)), 112)
        self.assertFalse(lib.t_tlsf_malloc(t, 1 << 40))         # far too big: null, no crash
        self.assertFalse(lib.t_tlsf_malloc(t, (1 << 64) - 1))
        self.assertFalse(lib.t_tlsf_realloc(t, a, 1 << 40))     # failed realloc keeps the block
        self.assertEqual(lib.t_tlsf_block_size(a), 16)
        lib.t_tlsf_free(t, None)                                # free(null) does nothing
        self.assertFalse(lib.t_tlsf_realloc(t, a, 0))           # realloc(p, 0) frees
        self.check("exact")
        # a request that fits only after freed neighbours merge
        lib.t_tlsf_init(t, heap.lo, 1 << 16)
        blocks = [lib.t_tlsf_malloc(t, 1000) for _ in range(40)]
        self.assertTrue(all(blocks[:60]))
        for b in blocks[10:30]:
            lib.t_tlsf_free(t, b)
        self.check("merged")
        big = lib.t_tlsf_malloc(t, 19 * 1016)
        self.assertTrue(big)
        self.assertTrue(blocks[10] <= big < blocks[30])         # it reused the merged hole
        self.check("reused")

    def test_create_in_one_buffer(self):
        lib = self.lib
        for misalign in (0, 1, 8, 15):
            heap = Heap(1 << 16, misalign)
            t = lib.t_tlsf_create(heap.lo, 1 << 16)
            self.assertTrue(t)
            self.assertEqual(t % 16, 0)
            self.assertEqual(lib.t_tlsf_check(t), 0)
            p = lib.t_tlsf_malloc(t, 5000)
            self.assertTrue(t + lib.t_sizeof_tlsf() <= p < heap.hi)
            self.assertFalse(lib.t_tlsf_create(heap.lo, 100))     # too small for the control block


class Arena(unittest.TestCase):
    def test_random(self):
        lib = load_lib()
        a = np.zeros(lib.t_sizeof_arena(), np.uint8).ctypes.data
        rng = np.random.default_rng(3)
        for misalign in (0, 3):
            heap = Heap(20000, misalign)
            lib.t_arena_init(a, heap.lo, 20000)
            live, marks, used, ident, peak = [], [], 0, 0, 0
            for step in range(4000):
                what = f"step {step}"
                r = rng.random()
                if r < 0.8:
                    size = int(rng.integers(0, 600))
                    align = 1 << int(rng.integers(0, 9))
                    if rng.random() < 0.03:
                        align = int(rng.choice([0, 3, 24, 100]))             # not powers of two: null
                    p = lib.t_arena_alloc(a, size, align)
                    cur = heap.lo + used
                    fits = align and not align & (align - 1) and (-cur) % align + size <= 20000 - used
                    self.assertEqual(bool(p), bool(fits), what)
                    if p:
                        self.assertEqual(p % align, 0, what)
                        self.assertEqual(p, cur + (-cur) % align, what)       # nothing wasted but alignment
                        used = p - heap.lo + size
                        peak = max(peak, used)
                        if size:
                            ident += 1
                            heap.claim(self, ident, p, size, what)
                            live.append((p, size, used))
                elif r < 0.9:
                    marks.append(lib.t_arena_mark(a))
                    self.assertEqual(marks[-1], used)
                elif marks:
                    m = marks.pop()
                    lib.t_arena_reset_to(a, m)
                    used = m
                    while live and live[-1][2] > m:
                        p, size, _ = live.pop()
                        heap.release(p, size)
                if step % 50 == 0:
                    heap.verify_all(self, what)
            self.assertEqual(lib.t_arena_peak(a), peak)


class Pool(unittest.TestCase):
    def test_random(self):
        lib = load_lib()
        pool = np.zeros(lib.t_sizeof_pool(), np.uint8).ctypes.data
        rng = np.random.default_rng(4)
        for block, align, misalign in ((24, 16, 0), (8, 8, 5), (100, 32, 7), (1, 1, 1)):
            with self.subTest(block=block, align=align):
                heap = Heap(10000, misalign)
                n = lib.t_pool_init(pool, heap.lo, 10000, block, align)
                al = max(align, 8)
                bs = -(-max(block, 8) // al) * al
                self.assertEqual(n, (10000 - (-heap.lo) % al) // bs)
                live, ident = {}, 0
                for step in range(3000):
                    what = f"block {block} step {step}"
                    if not live or rng.random() < 0.55:
                        p = lib.t_pool_alloc(pool)
                        self.assertEqual(bool(p), len(live) < n, what)        # fails exactly when full
                        if p:
                            self.assertEqual(p % al, 0, what)
                            self.assertTrue(lib.t_pool_owns(pool, p))
                            ident += 1
                            heap.claim(self, ident, p, bs, what)
                            live[ident] = p
                    else:
                        k = list(live)[int(rng.integers(len(live)))]
                        p = live.pop(k)
                        heap.release(p, bs)
                        lib.t_pool_free(pool, p)
                    self.assertEqual(lib.t_pool_used(pool), len(live))
                    if step % 50 == 0:
                        heap.verify_all(self, what)
                if live:
                    p = next(iter(live.values()))
                    self.assertFalse(lib.t_pool_owns(pool, p + 1))
                self.assertFalse(lib.t_pool_owns(pool, heap.hi + 64))
        self.assertEqual(lib.t_pool_init(pool, heap.lo, 10000, 24, 12), 0)       # alignment not a power of 2


if __name__ == "__main__":
    unittest.main()
