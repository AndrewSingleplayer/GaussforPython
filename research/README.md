# Research: Gaussian splats on phones

What the renderers in this repository are based on, with the measurements behind each decision.

| File | Contents |
|---|---|
| [01-splat-math.md](01-splat-math.md) | the math: covariance, EWA projection, anti-aliasing, alpha, compositing, spherical harmonics, formats and their precision |
| [02-speed.md](02-speed.md) | what makes it fast, measured: tight shapes, sorting variants, data sizes, redraw policy, allocators |
| [03-quality.md](03-quality.md) | what makes it look wrong: aliasing (Mip-Splatting), popping (StopThePop), view-dependent colour, precision |
| [04-phones-and-webar.md](04-phones-and-webar.md) | what Safari on an iPhone allows, and AR without WebXR |
| [analyze_scenes.py](analyze_scenes.py) | the measurements on the six real captures (`python3 research/analyze_scenes.py`) |

## Findings that changed the code

1. **Tight shapes cut the pixel work 2–5×.** A splat can be cut where it adds less than 1/255,
   an ellipse of `sqrt(2 ln(255 o))` standard deviations. Covering that ellipse with an oriented
   rectangle takes 2.1–4.7× fewer pixels than v1's 3σ square on the real scenes. The web viewer
   draws that rectangle.
2. **f16 loses most small splats.** 78–93% of the splats in real captures have a variance below
   the smallest normal f16. The web format divides the covariance by its largest entry before
   converting (full f16 precision). The v1 `PackedSplat` still stores raw f16: to fix.
3. **Quantized quaternions need renormalizing.** Rounding can push `|xyz|` past 1. Without
   renormalizing, the covariance is up to 4% too large. HA++'s `quat_to_mat3` did it right; the
   first JavaScript fallback didn't (found by `tests/test_web.py`).
4. **Sorting: simple wins on this machine.** A 16-bit counting sort beat a branch-free variant
   and a two-pass radix sort (13 ms for 740k splats in WebAssembly). The radix sort may still win
   on a phone's smaller caches: that needs a phone.
5. **HA++ in WebAssembly vs JavaScript:** up to 3× faster for math (decode), the same for
   memory-bound work (sort).
6. **View-dependent colour is scene-dependent:** 16% of the colour variation for glossy
   figurines, 0.2% for a matte plush. SH degree 1 as an option is the next quality step.
7. **iPhone Safari has no WebXR.** AR in the browser uses the camera plus the motion sensors
   directly. Rotation tracking is done. Floor tracking (6DoF) is next, in HA++.

## Next

- v1 compute renderer: exact ellipse–tile test (1.2–3.8× fewer pairs to sort), scaled-f16 covariance.
- SH degree 1 option in the web format.
- Floor tracking (6DoF) for the web AR mode.
- Measurements on an actual iPhone: the fps the adaptive controller settles on, WebAssembly vs
  JavaScriptCore, radix vs counting sort.
