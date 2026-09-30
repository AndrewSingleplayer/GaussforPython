# Memory allocators in HA++

`happ/lib/mem.ha` (`import "mem.ha";`) has three allocators and a fast copy
routine, all written in HA++. This page covers why they exist, how they work
and how fast they are. The API is in the
[language reference](LANGUAGE.md#memory-allocators-happlibmemha).

## Why a renderer needs its own allocators

At 60 frames per second a frame has 16.7 ms. A general-purpose `malloc`
is fast on average but has no upper bound: now and then it asks the
operating system for memory (`mmap`, `brk`) or hands it back, and that one
call can take tens of microseconds or more. In a frame loop, one slow call
means a late frame.

The three allocators here never call the operating system. The host app
gives each one a block of memory once, and every operation after that has a
small, fixed worst-case cost.

| | Arena | Pool | TLSF |
|---|---|---|---|
| For | per-frame scratch data | many objects of one size | anything else (a general `malloc`) |
| Allocate | move a pointer | pop a list, or take the next never-used block | two bit scans + unlink |
| Free | all at once (`arena_reset`, or back to a mark) | push onto a list | merge with free neighbours + link |
| Overhead per allocation | 0 (only alignment padding) | 0 | 16 bytes |
| Fragmentation | none | none | low (good-fit, immediate merging) |

## How TLSF works

TLSF (Two-Level Segregated Fit, Masmano, Ripoll, Crespo and Real, ECRTS
2004) keeps free blocks in size-class lists and finds a fitting list with
bit instructions instead of a search:

- **First level:** the power of two of the size, `floor(log2(size))`.
- **Second level:** each power of two split into 32 equal ranges. Below 512
  bytes the lists are exactly 16 bytes apart.
- **Two bitmaps:** one bit per non-empty list. `malloc` rounds the request
  up to the start of the next list, so any block found there is big enough.
  It then finds the first non-empty list at or above it with two
  count-trailing-zeros instructions (`ctz`).
- **Immediate merging:** each block records its physical neighbour and
  whether that neighbour is free. `free` merges with the free blocks on
  both sides in O(1), so free memory doesn't stay chopped into small pieces.

| Parameter | HA++ | Conte's C TLSF |
|---|---|---|
| Alignment of every result | 16 bytes (vec4/SIMD friendly) | 8 bytes |
| Overhead per block | 16 bytes | 8 bytes |
| Lists | 24 × 32 (sizes up to 4 GiB) | 24 × 32 |
| Pools | up to 16 | any number (linked) |
| Self-check | `tlsf_check()`: links, flags, merged neighbours, list membership, bitmaps, byte count | `tlsf_check()` |

## Speed

`python3 bench/alloc_bench.py` replays the same request sequence through each
allocator. It builds everything with the same compiler and flags
(`clang -O3 -DNDEBUG -march=x86-64-v3`). Conte's TLSF v3.1 is downloaded
from GitHub; it has a BSD license and isn't stored here. Machine: 4-core
Intel Xeon VM at 2.1 GHz, glibc 2.39. `ns/op` is the whole run divided by
the number of operations. The percentiles are per-call latencies from the
CPU cycle counter. The latency run is repeated 5 times, and each column
keeps its lowest value.

**Mixed heap:** 16 B to 64 KB, random malloc/realloc/free, up to 8192 alive.

| allocator | ns/op | p50 | p99 | p99.9 |
|---|---|---|---|---|
| glibc malloc | 95.7 | 63 ns | 567 ns | 2173 ns |
| C TLSF (Conte) | 62.2 | 57 ns | 313 ns | **1529 ns** |
| HA++ TLSF | **59.9** | 58 ns | **241 ns** | 2095 ns |

**One frame:** 64 temporary buffers of 64 B to 256 KB, then all released.

| allocator | ns/op | p50 | p99 | p99.9 |
|---|---|---|---|---|
| glibc malloc | 156.9 | 49 ns | 2547 ns | 3587 ns |
| C TLSF (Conte) | 56.4 | 32 ns | 133 ns | 334 ns |
| HA++ TLSF | 42.5 | 26 ns | 44 ns | 134 ns |
| HA++ Arena | **11.2** | 0 ns | **7 ns** | **15 ns** |

**Nodes:** 32-byte objects, random alloc/free, up to 65,536 alive.

| allocator | ns/op | p50 | p99 | p99.9 |
|---|---|---|---|---|
| glibc malloc | 17.8 | 18 ns | 80 ns | 158 ns |
| C TLSF (Conte) | 16.9 | 23 ns | 100 ns | 190 ns |
| HA++ TLSF | 19.3 | 26 ns | 94 ns | 187 ns |
| HA++ Pool | **13.4** | **11 ns** | **47 ns** | **116 ns** |

What the numbers show:

- **HA++ TLSF vs the C TLSF:** about the same average speed. In the mixed
  and node tests the difference changes sign from run to run (±2 ns). HA++
  is faster in the frame test (42 vs 56 ns/op). Its `malloc` tail is about
  2× shorter (p99 181–217 ns vs 385 ns when the mixed test is split by
  operation).
- **Where the C TLSF wins:** the p99.9 of the mixed test. Those slowest
  calls are `realloc`s that move a large block, and the time goes into
  copying it. For large copies of memory that isn't in the cache, glibc's
  `memcpy` uses the x86 `rep movsb` instruction, which skips reading the
  destination before overwriting it. HA++ libraries don't link the C
  library and have no inline assembly, so they can't use it. ARM64 phones
  have no `rep movsb`; their system `memcpy` is built from load/store-pair
  loops, the kind of code HA++ generates.
- **glibc in the frame test:** a p99 of 2.5 µs, because blocks above 128 KB
  come from `mmap` and go back with `munmap`, which are system calls. TLSF
  and the Arena never make one.
- **Arena for per-frame data:** 4× faster than any general allocator, with
  a p99.9 of 15 ns. The splat renderer's per-frame buffers belong here.
- **Maximum latency** is not in the tables. On this virtual machine every
  allocator, glibc included, shows a few calls of 25–200 µs per run. Those
  are timer interrupts and the hypervisor, not the allocator: they appear
  even for the Arena, whose allocation is three instructions.

### The copy routine (`mem_copy`)

TLSF blocks are 16-byte aligned. On x86 a 32-byte vector load or store at a
16-byte boundary straddles two cache lines half the time. So a plain copy
loop ran at half the speed of `memcpy` for 4–16 KB. `mem_copy` first
copies up to 31 bytes, so the destination reaches a 32-byte boundary. After
that, no store crosses a cache line. Measured copy time (best of 300):

| size | plain loop | `mem_copy` method | glibc `memcpy` |
|---|---|---|---|
| 4 KB | 102 ns | 49 ns | 57 ns |
| 16 KB | 309 ns | 118 ns | 119 ns |
| 64 KB | 1769 ns | 1498 ns | 1433 ns |

## How it is tested

- **`tests/test_alloc.py`** runs random operation sequences against a model
  kept in Python. After every operation it checks that results are aligned,
  inside the managed memory and not overlapping a live block. It checks
  that every live block's bytes are intact, that `realloc` kept the old
  contents, and that `malloc` failed only when no free block was big
  enough. `tlsf_check()` must also report all invariants intact. It covers
  small heaps (out-of-memory paths), a 2 MB heap, two pools, misaligned
  memory, `mem_copy` at all 64 × 64 alignments, and the Arena and Pool.
- **`tests/mutate_alloc.py`** plants 22 typical allocator bugs, one at a
  time: a missed merge, a flag not cleared, an off-by-one in the split
  threshold, a copy that is too short, a bitmap bit left set, and others.
  It checks that the tests fail for each. All 22 are caught.

## Limits

- Not thread-safe: give each thread its own allocator. That is the usual
  design in game engines: per-thread arenas and pools, no locks.
- Measured on an x86 PC. Speed on a phone's ARM64 cores hasn't been
  measured on a device yet.
- TLSF blocks up to 4 GiB each (more memory is fine through several pools).
