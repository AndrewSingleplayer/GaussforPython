# To be added: 4D Gaussian splat animation

**Status: not built. This is the plan.**

## Goal

Play a finished Gaussian splat animation (display only), in real time, in the web viewer and in
AR. The animation is made elsewhere and loaded into the page. Nothing is edited or captured live.
Placing, tracking, light matching and touch work the same as for a still scene; only the splats
move.

## What it builds on

| Already there | How the animation uses it |
|---|---|
| The 16-byte download format and its decoder (`web/pack.py`, `web/splatweb.ha`) | colour, opacity and scale are stored once, and only the movement per key frame |
| The sort in a background thread (HA++ → WebAssembly) | runs every frame while the animation plays, not only when the camera moves |
| Edits made while drawing (the light matching in `web/src/viewer.js`) | the GPU blends between key frames in the same place, without changing the data |
| AR placement and tracking (`web/src/ar.js`, `web/src/tracker.mjs`) | unchanged |

## Input formats

| What the animation file is | What playback needs | Effort |
|---|---|---|
| **A sequence of `.ply` files**, one per frame (the most common export) | pack them into one file: what doesn't change once, then only the movement per key frame; blend between key frames | medium; the file size is the limit |
| **Splats with built-in motion**: each splat carries its own path over time (Spacetime Gaussians) | compute each splat's position for the current moment while drawing | medium; small files |
| **A trained network that moves the splats** (most 4DGS research code) | convert it ("bake" it) on a PC into one of the two above first | a conversion step on top |
| **Splats attached to a skeleton** | store only the bone movements per frame | medium to hard; needs a rigged source |

Start with the `.ply` sequence: it is what most tools export.

## File format (plan)

- **Header:** the splat count, the number of key frames, key frames per second, and the bounding
  box over all frames.
- **Still part, once:** the current 16 bytes per splat, taken from the first frame.
- **Moving part, per key frame:** 10 bytes per moving splat.
  - position: 3 x 16 bits, relative to the bounding box;
  - rotation: 4 x 8 bits.
- **Only splats that move are stored per key frame.** A list says which they are, so splats that
  never move cost nothing after the first frame.
- **Key frames:** 5-10 per second. In between, the GPU blends them: positions linearly, rotations
  normalised.
- **Order:** key frames are stored in time order, so playback can start as soon as the first
  second has arrived.

## Size budget

A 50,000-splat object, in a 10-second loop:

| Stored as | Size |
|---|---|
| every frame in full, 30 fps (16 bytes x 50,000 x 300) | about 240 MB: too big for a phone |
| the movement only, 5 key frames per second (10 bytes x 50,000 x 50) | about 25 MB |
| the same, with 30% of the splats moving | about 8 MB |

Downloads and Safari's memory limit break first, not speed. Effort to make the files smaller is
worth more than effort to make the code faster.

## Speed (measured)

Measured on the 4-core x86-64 virtual machine, in Node 22 (V8). A phone hasn't been measured. The
work runs in the background thread, so drawing doesn't wait for it.

| Splats | Blend positions between key frames | Re-sort | Total | 60 fps (16.7 ms)? |
|---|---|---|---|---|
| 50,000 | 0.24 ms | 0.9 ms | about 1 ms | easily |
| 345,000 | 1.8 ms | 6.2 ms | about 8 ms | yes |
| 865,000 | 4.5 ms | 14 ms | about 18.5 ms | just under; 30 fps is easy |

The GPU draws a moving splat at the same cost as a still one. The AR tracking has its own thread
(about 2 ms per camera frame), so the two don't compete. The sort times are from
[AR.md](AR.md#speeds).

## Playback

1. **Load:** download the still part, then the key frames in order, with a progress bar. Start
   playing once the first second is in.
2. **Draw:** the two key frames around the current time sit in GPU textures. The vertex shader
   blends them, the same way the light matching edits splats.
3. **Sort:** the background thread blends the same positions and re-sorts every frame while the
   animation plays. A large scene can sort every second frame: the order changes little in 33 ms.
4. **Controls:** play/pause, loop and speed. In AR the scene is placed, slid, resized and turned
   as now. The contact shadow uses the first frame's footprint.

## Later: the GPU path (HA++ → WebGPU)

For millions of moving splats, the blend and the sort move to the GPU. HA++ already has a GPU
radix sort, running on Metal and Vulkan. Safari has WebGPU since iOS 26. What it takes:

1. **A WGSL output in `happ/gpu.py`, next to GLSL and Metal.** The points that need care:
   - 3-component vectors are stored as three floats, as for Vulkan;
   - kernel parameters go in a uniform buffer, because WebGPU has no push constants;
   - buffers used with atomics are declared as atomic;
   - f16 needs the `shader-f16` extension, or falls back to f32;
   - integer division keeps HA++'s rule that every input gives a defined result.
2. **A small JavaScript runtime for WebGPU** (buffers, kernels, dispatches), with the loader
   generated by HA++.
3. **Tests:** run the kernels in headless Chromium's WebGPU and compare them with the CPU, as the
   fuzzer does for Vulkan and Metal.

Limits of this path:
- **Two paths to maintain.** Older iPhones have no WebGPU, so the WebGL2 path stays.
- **The AR page would draw with WebGPU too,** because WebGL2 and WebGPU can't share data.
- **The best drawing method isn't known yet.** Apple GPUs blend quads cheaply. The likely best mix
  is HA++'s GPU sort plus quad drawing in WebGPU, not HA++'s compute rasterizer. Measure it on an
  iPhone first.

## Tests (plan)

- **Packing round trip:** pack a `.ply` sequence, then decode and blend it. Check against NumPy
  at several times, including between key frames.
- **Sort:** sorting while playing gives the same order as sorting the blended positions with
  NumPy.
- **The page:** in headless Chromium the animation loads, plays and loops, and can be placed in
  AR with no script errors. This joins `tests/test_browser.py`, which runs before every deploy.
- **Sizes:** check the file size against the budget above for a test animation.

## Others that already play animated splats on the web

Animated splats in a browser are not new:
- **[splaTV](https://radiancefields.com/splatv-dynamic-gaussian-splatting-viewer)** plays Spacetime
  Gaussians in the browser.
- **[SuperSplat 1.13](https://www.cgchannel.com/2025/01/superspat-1-13-plays-back-animated-4d-gaussian-splats/)**
  plays numbered sequences of `.ply` files.
- **[Chronosplat](https://radiancefields.com/chronosplat-plays-4d-gaussian-splat-sequences-from-static-files-in-the-browser)**
  streams compressed frames as a flipbook.

What is less common is animated splats in AR on an iPhone, in Safari, with the page's own tracking
and no WebXR. Before calling it a first, search again and test it on several phones.

## Open questions

- Which tool makes the animations this is for, and what does it export?
- How long are they, how many splats, and how much of each scene moves?
