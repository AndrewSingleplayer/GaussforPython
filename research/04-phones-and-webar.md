# 4. Phones, Safari and AR without WebXR

## What an iPhone gives a web page (Safari, 2026)

| Feature | Status | What it means here |
|---|---|---|
| WebGL2 | every iPhone on iOS 15+ | the web viewer draws with it: vertex/fragment shaders, instancing, integer textures |
| WebGPU | on by default since Safari 26 / iOS 26 | compute shaders in the browser; the v1 compute pipeline could run there (HA++ would need a WGSL back end) |
| WebAssembly with SIMD128 | Safari 16.4+ | HA++'s `web-wasm32` target (decode, cull, sort) |
| Threads (SharedArrayBuffer) | only on pages served with cross-origin isolation headers | one Web Worker instead |
| **WebXR (immersive-ar)** | **not supported**; Apple hasn't announced it for iPhone | no ARKit tracking through the browser |
| Camera (`getUserMedia`) | yes, on https pages, after a permission prompt | the AR background |
| Motion sensors (`DeviceOrientation`/`DeviceMotion`) | yes, after `requestPermission()` from a tap | rotation tracking |
| AR Quick Look (USDZ) | yes, but it doesn't render Gaussian splats | not usable for splats; only native RealityKit apps can draw them |

Apple GPUs are tile-based deferred renderers: blending happens in on-chip tile memory. That is
why drawing splats as blended quads (the web viewer) is cheap there, and why the render
resolution and the overdraw decide the frame rate.

## AR without WebXR: what is possible in a browser

The browser gives the camera image and the orientation sensors. Everything else has to be done
by the page's own code:

| Level | What it tracks | How | Effort | Here |
|---|---|---|---|---|
| **Look-around (3DoF)** | rotation | fused `DeviceOrientation` angles turn the virtual camera; the camera image fills the background | small | **done**: `web/src/ar.js` |
| Image target | position and rotation relative to a printed picture | feature matching + homography + pose (PnP) per frame | medium | not planned |
| **Floor tracking (6DoF)** | position and rotation relative to the floor | track corner features on the floor (optical flow), estimate the floor plane's motion between frames, fuse with the gyroscope; scale from the phone's height above the floor | large | **next**: in HA++, compiled to WebAssembly |
| Full SLAM | a map of the room | what ARKit does | very large | not planned |

The 3DoF mode uses these conventions:
- the W3C rotation `R = Rz(α) Rx(β) Ry(γ)`, corrected for the screen's rotation;
- the camera looks along the device's −z;
- a field of view matching the iPhone main camera: 26 mm equivalent, about 67° across the long
  side of the image;
- the scene placed once, where the phone looks when AR starts.

`web/src/ar.js` has this math. It was checked in Node (turning left moves the scene right,
tilting down moves it up, landscape stays consistent), and in headless Chromium with a fake
camera and simulated orientation events.

What the 3DoF mode can't do: when you walk, the scene keeps its distance to you. That is the
difference floor tracking makes.

8th Wall, the main commercial browser SLAM, shut down its hosted service in February 2026. Its
open-source release doesn't include the SLAM engine (that exists only as a binary under its own
license). A browser AR engine with its own tracking is therefore something to build, not
something to call.

## Hosting

The claude.ai link view blocks the camera for every page. The AR page therefore has its own
address on GitHub Pages, built by `.github/workflows/pages.yml`. The 3D viewer without AR works
in both places.

## Budgets from other viewers

- Spark (World Labs) suggests 1–3 million splats for iPhones and 1–2 million for Android phones.
- Large files fail on Safari because of memory: a 885 MB, 3-million-splat file didn't load on an
  iPhone 13. Compact formats matter. Here the largest scene is 13.8 MB to download and 28 MB on
  the GPU.

## Sources

- WebXR on iOS Safari: https://www.browserstack.com/guide/webxr-and-compatible-browsers ,
  https://en.wikipedia.org/wiki/WebAR
- WebGPU in Safari 26: https://appdevelopermagazine.com/webgpu-in-ios-26/
- AR Quick Look and Gaussian splats: https://developer.apple.com/forums/thread/804604
- 8th Wall shutdown and open source: https://www.roadtovr.com/?p=126602 ,
  https://8thwall.org/blog/8th-wall-open-source
- Spark performance: https://sparkjs.dev/docs/performance/
- Safari memory with large splat files: https://forum.playcanvas.com/t/solved-large-3d-gaussian-splatting-file-doesnt-load-on-mobile/38758
- Apple GPUs and tile-based deferred rendering:
  https://developer.apple.com/documentation/metal/tailor-your-apps-for-apple-gpus-and-tile-based-deferred-rendering
