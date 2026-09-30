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
| **Floor from gravity** | where the floor is, at what scale | the floor is the horizontal plane (gravity from the motion sensors) a standing person's phone height below the camera; a ray from the screen meets it | small | **done**: ring on the floor, tap to place, contact shadow |
| Image target | position and rotation relative to a printed picture | feature matching + homography + pose (PnP) per frame | medium | not planned |
| **Floor tracking (6DoF)** | position and rotation relative to the floor | track corner features on the floor (optical flow), solve the camera pose from them, fuse with the gyroscope; scale from the phone's height above the floor | large | **done**: `web/track.ha` (HA++ → WebAssembly) + `web/src/tracker.mjs` |
| Full SLAM | a map of the room | what ARKit does | very large | not planned |

The 3DoF mode uses these conventions:
- the W3C rotation `R = Rz(α) Rx(β) Ry(γ)`, corrected for the screen's rotation;
- the camera looks along the device's −z;
- a field of view matching the iPhone main camera: 26 mm equivalent, about 67° across the long
  side of the image;
- the floor 1.35 m below the phone (a phone held in front of a standing adult); the scene is
  placed where a tap's ray meets the floor, standing up, facing the camera, 0.7 m tall.

`web/src/ar.js` has this math. It was checked in Node (turning left moves the scene right,
tilting down moves it up, landscape stays consistent), and in headless Chromium with a fake
camera and simulated orientation events.

What the 3DoF mode can't do: when you walk, the scene keeps its distance to you. That is the
difference floor tracking makes.

## Floor tracking

Every tracked point lies on the floor, a plane at a known height below the starting camera. So a
new point's 3D position is where its pixel's ray meets the floor. It needs no second view, no
triangulation and no initialisation motion. This is the idea behind "instant placement" in
commercial web AR. Per camera frame (202 x 360 pixels, portrait):

1. **Predict** the rotation from the gyroscope's change since the last frame, and the position
   from the last two frames.
2. **Whole-image shift:** high-pass the quarter-size image and search the shift that best aligns
   it with the last one (mean absolute difference, ±28 pixels around the prediction, refined to a
   fraction of a pixel). If it disagrees with the prediction, the prediction was wrong (sensor
   timing, a glitch), and every point's starting position is corrected. PTAM's "small blurry
   image" is the same idea.
3. **Optical flow:** pyramidal Lucas-Kanade (4 levels, 11 x 11 windows) with a brightness offset
   per window, so exposure changes don't move points. It is checked by tracking back (at most
   0.7 pixels of disagreement allowed).
4. **Pose:** Gauss-Newton on the reprojection error with a Huber loss. It runs once from the image
   alone. Then it runs again without the points that don't fit, with the gyroscope's rotation as a
   prior, but only if that rotation agrees with the image within 0.6°. This two-step solve
   matters. With a robust loss each point's pull is capped, so a wrong prior (a gyroscope reading
   40 ms off during a fast turn) can outweigh 100 good points. A right one steadies what the image
   tells least well: turning vs. moving sideways when the floor points are far away.
5. **Map:** points that don't fit for three frames are dropped (they are on furniture, or were
   followed wrongly). New points are added in empty cells of a 16-pixel grid, on floor at least
   11.5° below the horizon and at most 6 m away.

**Timing.** A camera frame reaches the page 50-100 ms after the motion sensor readings from the
same moment. Reading the gyroscope at the frame's arrival time makes the prediction wrong by
several degrees in a fast turn. The tracker keeps 2.5 s of readings and of its own rotations.
Every 10 frames it looks for the delay (0-200 ms) that makes the gyroscope's turning between
frames match the image's turning. It takes the median of the last 7 clear answers.

**Measured on simulated walks** (`tests/track_sim.py`). The simulator renders a textured floor
for a moving camera: 2x supersampled, noise, exposure drift, optional motion blur, rolling
shutter, boxes standing on the floor. The simulated gyroscope has delay, drift and noise. The
anchor error is how far from the real floor spot a point placed at the start is drawn, in
tracking-image pixels (about 2.4 screen points each on an iPhone):

| Walk | Floor | Anchor error, median / max | Position error at the end |
|---|---|---|---|
| 150° around a spot, 4 m | terrazzo | 0.34 / 0.83 px | 0.8 cm |
| full circle, 20 s, 9.9 m | terrazzo / tiles / wood | 0.7 / 2.2, 1.6 / 3.5, 2.5 / 3.9 px | 2.5 / 5.4 / 5.1 cm |
| two 70° turns at up to 275°/s, camera 100 ms late | terrazzo | 0.6 / 2.5 px | 1.4 cm |
| the same, camera 20 ms late | terrazzo | 0.8 / 3.8 px | 2.0 cm |
| rolling shutter (20 ms) + 40° turns | wood / carpet | 1.6 / 3.3, 2.8 / 6.9 px | 6.6 / 9.4 cm |
| phone 1.15 m high (1.35 assumed), lens 72° (67 assumed) | terrazzo | 1.5 / 2.8 px | 1.2 cm (after scaling by the height ratio) |

The measured delay converges to the true one (20, 50, 100 ms). Each frame takes 1.5-2.5 ms in
Node (WebAssembly), 1-5 ms in headless Chromium. In the browser, with a synthetic camera video
(`--use-file-for-fake-video-capture`), the page's tracker followed a 1.0 m slide to within 1 cm.

What it doesn't do yet:
- **Relocalization.** After tracking is lost (camera covered, blank floor, very fast motion), the
  scene keeps its place relative to the last known position. It can end up shifted by however far
  the phone moved while lost. Recognising the old floor points again (keyframes and descriptors)
  is the next step.
- **Real devices.** These numbers are from simulations and a desktop browser. The camera delay,
  the lens and Safari's frame timing on an actual iPhone are measured by the tracker itself as it
  runs; the page shows them.

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
