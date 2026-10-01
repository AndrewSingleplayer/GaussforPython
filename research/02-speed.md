# 2. Making splats fast: algorithms, measured

Every number here was measured in this repository, on the six real captures
([`analyze_scenes.py`](analyze_scenes.py)) or with the benchmarks named in each section. The
machine is a 4-core x86-64 VM with no GPU. GPU-side numbers are therefore work counts (pixels,
pairs, bytes), which carry over to a phone, not times, which don't.

## Where the time goes

A frame has four kinds of work:

| Work | Scales with | Main lever |
|---|---|---|
| Per splat: project, cull, colour | number of splats | cull early, store compactly, draw fewer splats |
| Sort | visible splats (or splat-tile pairs) | fewer items, cache-friendly sort, sort less often |
| Per pixel: blend | covered pixels × splats per pixel (overdraw) | tighter shapes, lower resolution |
| Memory traffic | bytes per splat × splats | smaller formats |

On phones the pixel work and the memory traffic dominate; the per-splat arithmetic is cheap.

## Tighter shapes: 2–5× fewer pixels

At an iPhone-like render size (590×1280, a 50° vertical field of view, each scene framed as in
the orbit animation):

| scene | splats | on screen | overdraw, 3σ square (v1) | overdraw, tight quad | tile pairs v1 | tile pairs, exact ellipse test |
|---|---|---|---|---|---|---|
| racoons | 932,560 | 708,441 | 142× | 67× | 1,757,764 | 1,426,138 |
| lizard | 786,233 | 588,854 | 66× | 29× | 1,209,188 | 1,014,124 |
| halo | 345,217 | 124,010 | 362× | 92× | 1,613,844 | 614,670 |
| firepit | 561,997 | 359,291 | 561× | 153× | 2,801,608 | 1,209,014 |
| unicorn | 49,602 | 42,560 | 435× | 115× | 1,618,362 | 602,333 |
| skull | 162,598 | 110,463 | 1103× | 237× | 4,049,466 | 1,076,342 |

"Overdraw" is the total area of all drawn shapes divided by the screen area.

- **Tight quad** is a rectangle along the ellipse's axes, cut where the splat's contribution
  falls below 1/255 (half-axes `k√λ`, `k = sqrt(2 ln(255 o))`, see
  [`01-splat-math.md`](01-splat-math.md)). It covers 2.1–4.7× fewer pixels than the v1 bounding
  square. The gain is largest for scenes with many large, faint splats. The web viewer draws
  exactly this quad.
- **Exact ellipse–tile test**: for the compute rasterizer, testing each 16×16 tile against the
  ellipse (instead of the square's tile rectangle) cuts the pairs to sort by 1.2–3.8×.
  FlashGS and Speedy-Splat report the same effect. This is the main change planned for the v1
  compute renderer (`gaussian/splat.ha`).
- Opacity alone is no filter: only 0–0.9% of the splats are below 1/255 in these captures.

## Sorting

The web viewer sorts on the CPU, in a Web Worker, by depth: a 16-bit key per visible splat.

| raccoons: 740k visible of 864k (best of 30, one core, Node / V8) | time |
|---|---|
| HA++ → WebAssembly, cull + 16-bit counting sort (what ships) | 13.0–14.5 ms |
| the same, cull loop without branches | 16.0 ms |
| the same, two 8-bit LSD radix passes instead of one 65,536-bucket counting sort | 13.8 ms |
| JavaScript (same algorithm) | 12.2–14.4 ms |

- Every variant produces the identical order. Both sorts are stable over the same keys.
- **A branch-free cull loop is slower here.** From these cameras most splats are visible, so
  the branch predicts well, and writing every element costs memory traffic.
- **Radix sort didn't beat the counting sort** on this machine, whose CPU has 266 MB of cache:
  the counting sort's scattered writes still hit in cache. A phone's much smaller caches may
  favour the radix version. That needs a measurement on a phone.
- **WebAssembly vs JavaScript:** once V8 has warmed up, both run at the same speed, for the sort
  (memory-bound) and for decoding (`exp`, quaternion → matrix) alike: V8 compiles a plain
  typed-array loop about as well as LLVM does. WebAssembly is faster on a page's first, small
  load, which JavaScript runs before it has warmed up. In a fresh process the unicorn (50k splats)
  decodes in 7-8 ms against 23-34 ms, which is the "up to 3×" first reported here. The raccoons
  (865k) take 120-137 ms in both. All six scenes: [`../docs/AR.md`](../docs/AR.md#speeds).
  Safari's JavaScript engine (JavaScriptCore) has to be measured on the phone itself.
- **Sorting less:** the sort runs only when the camera moves, and drawing never waits for it. It
  uses the last order, as every web splat viewer does.

The v1 compute pipeline sorts `(tile, depth)` pairs on the GPU instead: 4 passes of an 8-bit
radix sort over 1–4M pairs. There the lever is the pair count (table above).

## Smaller data

| | bytes per splat | raccoons, 864k splats |
|---|---|---|
| 3DGS `.ply` with SH3 | 248 | 214 MB |
| download format `.hspl` (web) | 16 | 13.8 MB |
| GPU format (web) | 32 | 27.7 MB of texture |

Quantization error of the 16-byte download format, by construction:
- position: 1/65535 of the bounding box. For a 2 m scene that is 0.03 mm.
- scale: steps of `lrange/255` in log space, about 6% per step (the same as SPZ).
- rotation: 8 bits per component, about 0.5°.

`tests/test_web.py` checks the decoder against NumPy. The decoded covariance is within 0.1% of
its largest entry.

## Doing less when nothing changes

- **Redraw only on change.** The web viewer draws a frame only after the camera moved, a new
  sort arrived, or the size or settings changed. An idle viewer uses no GPU.
- **Adaptive quality.** The viewer measures fps every second. Below 42 fps it first lowers the
  resolution, then draws fewer splats. Splats are stored most important first (opacity × area),
  so "fewer splats" means dropping the faintest and smallest ones. Above 57 fps it raises both
  again. On a phone with a 3× screen it starts at 1.5× resolution, not 3×: soft splats barely
  change at the higher resolution, and the pixel count is 4× smaller.

## Memory allocation in the frame loop

HA++'s allocators ([`../docs/MEMORY.md`](../docs/MEMORY.md)), measured against glibc:
- A **per-frame arena** takes 11 ns per allocation, with a p99.9 of 15 ns.
- glibc `malloc` in the same pattern takes 157 ns, with a p99 of 2.5 µs, because large blocks
  come from system calls.
- The TLSF allocator matches a C TLSF (42 vs 56 ns/op in the frame pattern) with a shorter
  `malloc` tail.

## What this points to next

1. The v1 compute renderer: an exact ellipse–tile test (1.2–3.8× fewer pairs) and the f16
   scaling fix from [`01-splat-math.md`](01-splat-math.md).
2. View-dependent colour (SH degree 1) for scenes that need it (see
   [`03-quality.md`](03-quality.md)).
3. Measure on an actual iPhone: JavaScriptCore vs WebAssembly, radix vs counting sort, the fps
   the adaptive controller settles on.

## Sources

- FlashGS: efficient 3D Gaussian splatting for large-scale and high-resolution rendering,
  https://arxiv.org/abs/2408.07967
- Speedy-Splat: fast 3D Gaussian splatting with sparse pixels and sparse primitives,
  https://arxiv.org/abs/2412.00578
- Spark (World Labs) performance notes: https://sparkjs.dev/docs/performance/
- PlayCanvas splat performance: https://developer.playcanvas.com/user-manual/gaussian-splatting/building/performance/
