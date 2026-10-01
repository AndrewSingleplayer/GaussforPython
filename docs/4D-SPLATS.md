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
| The 16-byte download format and its decoder (`web/pack.py`, `web/splatweb.ha`) | the splats are stored once; after that, only the groups' changes per frame |
| The sort in a background thread (HA++ → WebAssembly) | runs every frame while the animation plays, not only when the camera moves |
| Edits made while drawing (the light matching in `web/src/viewer.js`) | the GPU moves each splat by its groups in the same place, without changing the data |
| AR placement and tracking (`web/src/ar.js`, `web/src/tracker.mjs`) | unchanged |

## Input formats

| What the animation file is | What playback needs | Effort |
|---|---|---|
| **A sequence of `.ply` files**, one per frame (the most common export) | find groups of splats that move together and store only each group's change per frame (below) | medium |
| **Splats with built-in motion**: each splat carries its own path over time (Spacetime Gaussians) | compute each splat's position for the current moment while drawing | medium; small files |
| **A trained network that moves the splats** (most 4DGS research code) | convert it ("bake" it) on a PC into a `.ply` sequence first | a conversion step on top |
| **Splats attached to a skeleton** | the bones are the groups; store only their changes | medium |

Start with the `.ply` sequence: it is what most tools export.

**Which sequences the group method fits.** Each splat has to keep its identity across frames:
splat 7 in frame 1 is splat 7 in frame 2. 4DGS methods that move one fixed set of splats export
it that way. Sequences trained frame by frame, where every frame has its own splats, don't. Those
need the splats matched between frames first, or they fall back to storing each splat's movement.

## How the motion is stored: groups and deltas (plan)

Neighbouring splats move together. A hand's splats all follow the hand, and a lid's splats all
follow the lid. So the motion is stored per group, not per splat. The idea came from xplor3d. In
research, SC-GS (Sparse-Controlled Gaussian Splatting, CVPR 2024) drives all splats from a few
hundred control points the same way.

1. **Find the groups** on a PC, while packing. Group the splats by how they move over the whole
   animation, not only by where they are: two parts that touch but move differently end up in
   different groups. That gives a few hundred control points.
2. **Bind each splat once:** up to 4 nearest control points and their weights, 12 bytes per
   splat, stored once. Splats near a joint blend between the groups, so there is no seam.
3. **Per frame, store only the deltas:** for each control point, its change in rotation,
   position and colour (brightness and tint) since the last frame. That is about 10 bytes. A group
   that didn't move stores nothing.
4. **A full frame every second,** like the key frames in video. Without it, rounding errors would
   add up over a chain of deltas, and you couldn't jump into the middle of a chain. With it, the
   loop can restart and a timeline can seek.
5. **Check while packing.** The packer measures how far each splat lands from its real position in
   every frame. A splat that misses by more than a limit (say 2 mm, or half its own size) gets its
   own track. If many miss, the packer adds control points.
6. **What groups don't fit:** fire, smoke, liquid, and splats that appear or fade out. These use
   per-splat tracks, or a per-group opacity change.

## Size budget

A 30,000-splat object, in a 10-second loop at 30 fps:

| Stored as | Size |
|---|---|
| every splat, every frame (16 bytes x 30,000 x 300) | about 144 MB: too big for a phone |
| every splat's movement, 5 key frames per second (10 bytes x 30,000 x 50) | about 15 MB |
| **groups and deltas:** 500 control points, every frame | **about 2.5 MB** |

The 2.5 MB is made up of:
- the splats once: 480 KB;
- their bindings: 360 KB;
- a full frame every second: 80 KB;
- the deltas, 10 bytes x 500 x 290 frames: 1.5 MB.

It is less when parts stand still, because still groups store nothing.

## Speed (measured)

Measured on the 4-core x86-64 virtual machine, in Node 22 (V8). A phone hasn't been measured. The
work runs in the background thread, so drawing doesn't wait for it.

The GPU draws each splat from all 4 of its control points, in the vertex shader. The CPU needs the
positions only for the sort, and the sort only needs the order. The nearest control point almost
always gets that right, so the sort uses it alone:

| Splats | Positions for the sort (nearest control point) | Re-sort | Total | Fits |
|---|---|---|---|---|
| 50,000 | 0.6 ms | 0.9 ms | 1.5 ms | 60 fps, easily |
| 345,000 | 4.5 ms | 6.2 ms | 10.7 ms | 60 fps |
| 865,000 | 11.0 ms | 14 ms | 25 ms | 30 fps |

For comparison:
- **All 4 control points on the CPU:** 1.7 / 12.6 / 33 ms. That's exact, but more than the sort
  needs.
- **Per-splat key frames:** blending them takes 0.24 / 1.8 / 4.5 ms. That's cheaper on the CPU, but
  the files are 6 times bigger.

The AR tracking has its own thread (about 2 ms per camera frame), so the two don't compete. The
sort times are from [AR.md](AR.md#speeds).

## Playback

1. **Load:** download the splats and their bindings, then the frames in order, with a progress
   bar. Start playing once the first second is in.
2. **Draw:** each frame, apply the deltas to the control points (a few hundred transforms, done
   on the main thread). Put them in a small GPU texture. The vertex shader moves each splat by its
   4 control points and applies the colour change, the same way the light matching edits splats.
3. **Sort:** the background thread moves the splats by their nearest control point and re-sorts
   every frame while the animation plays. A large scene can sort every second frame: the order
   changes little in 33 ms.
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

- **Packing round trip:** pack a `.ply` sequence into groups and deltas, then play it back. Every
  splat must land within the packer's limit of its real position in every frame, checked with
  NumPy. This includes splats near joints and a jump into the middle of the loop.
- **Sort:** sorting by the nearest control point gives nearly the same order as sorting the
  exact positions. Measure how many pairs are swapped, and how far apart those are.
- **The page:** in headless Chromium the animation loads, plays and loops, and can be placed in
  AR with no script errors. This joins `tests/test_browser.py`, which runs before every deploy.
- **Sizes:** check the file size against the budget above for a test animation, with and
  without still parts.

## Others that already play animated splats on the web

Animated splats in a browser are not new:
- **[splaTV](https://radiancefields.com/splatv-dynamic-gaussian-splatting-viewer)** plays Spacetime
  Gaussians in the browser.
- **[SuperSplat 1.13](https://www.cgchannel.com/2025/01/superspat-1-13-plays-back-animated-4d-gaussian-splats/)**
  plays numbered sequences of `.ply` files.
- **[Chronosplat](https://radiancefields.com/chronosplat-plays-4d-gaussian-splat-sequences-from-static-files-in-the-browser)**
  streams compressed frames as a flipbook.

These are viewers built for desktop browsers. None of them does AR.

Animated splats on phones and in AR exist elsewhere, but in other forms:
- **Headset apps:** [Gracia](https://radiancefields.com/4d-gaussian-splatting) ships streamable
  4D splat apps on Quest and Apple Vision Pro. These are native apps, not phones and not a browser.
- **Research players for phones:** [DualGS](https://arxiv.org/pdf/2409.08353) and
  [4DGCPro](https://arxiv.org/html/2509.17513v2).
- **Skinned splat avatars:** they run on the web, on phones and in VR, and reach AR through WebXR
  ([arXiv 2510.13978](https://arxiv.org/html/2510.13978v2)). Safari on iPhone has no WebXR, so
  their AR doesn't run there.
- **Still splats in web AR on iPhone:**
  [8th Wall / Niantic Studio](https://8thwall.com/docs/studio/guides/gaussian-splats) supported
  them until it shut down in February 2026.

A short search (October 2026) found no 4D splat animation in AR in Safari on an iPhone, with the
page's own tracking and no WebXR. That may make this the first of its kind, but a short search
can't prove it. It also counts only once it is built and works reliably on several phones.

## Open questions

- Which tool makes the animations this is for, and what does it export?
- How long are they, how many splats, and how much of each scene moves?
