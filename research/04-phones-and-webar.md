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

**Camera image and scene in step.** The tracker's pose is for the frame it was given. Safari's
camera preview (a `<video>`) may already show a newer frame when the answer comes back: 33 ms
newer in Low Power Mode, 3° at a 90°/s turn, about 60 points on an iPhone screen. Bringing the
pose forward with the gyroscope removed the lag but made the scene jiggle. The hand shakes about
10 times a second, and any error in the guessed timing shows as the scene swimming against the
image. So the page draws the camera image itself, in WebGL. It shows the very frame the tracker
measured, with that frame's pose, as ARKit apps do; each tracked frame is copied into a texture
and shown when its answer arrives. Nothing needs timing, and the image shows about one screen
frame later than Safari's preview would. The pose is not smoothed. On simulated walks the pose's
own noise moves the scene less than 0.1 pixel from frame to frame, and a One Euro filter made it
worse, because the lag it adds follows the hand's shake. The camera is also asked for 60 fps. The
readout shows the camera's real frame rate.

**Light.** A capture carries the light of the room it was filmed in, usually bright and neutral.
In a dimmer or warmer room it looks pasted in. So, like ARKit's light estimation, the tracker
measures each camera frame's mean colour (`light_stats` in `track.ha`), and the colour of the
floor under the scene. The scene is then drawn to match. Its colours are scaled by the room's
brightness (relative to a normally exposed frame, to the power 0.75, between 0.45× and 1.2×) and
half-tinted by the room's colour cast. The bottom 8 cm darken toward the floor (contact
shading), and the bottom 30 cm pick up some of the floor's colour (bounce light). These are soft
edits made per splat in the vertex shader as it is drawn. The scene file and the decoded splats
are never changed, and "Light: original" shows them untouched.

**Placing and moving.** The first tap puts the scene on the ring, wherever the finger is. The
scene's base is the middle of its bottom slice (the part that touches the floor), so a leaning
scene still stands centred on the ring. Afterwards, two fingers slide it along the floor: the
floor points under the fingers' midpoint before and after the move give the shift. Pinch resizes
it, twist and one-finger drag turn it, and a tap moves it.

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

**Tables.** The tracker starts with the floor. When a table comes into view, its points are
taken for floor points at first, which is all a single view can say. So a new point is on
probation: it counts 5% in the pose until the phone has moved enough to see it from 3° further
round, and it still fits. A point on probation that misses by more than 1.5 px is freed at once.
It is then followed on as a free point, and its place is triangulated from its rays (the point
nearest to all of them, once they are 4° apart). A freed point that lands on the floor goes back
to being a floor point. Points 15 cm to 1.6 m above the floor that agree in height (8 or more
within 5 cm) make a surface, outlined by their convex hull. New points seen on it get their depth
at once, like floor points, so tracking holds when only the table is in view. The ring and the
tap go on the nearest surface under the middle of the screen, and a scene on a table stands
30 cm tall by default (0.7 m on the floor). Before placing, found surfaces are outlined faintly,
as ARKit apps do.

Measured (tests/track_sim.py --table): the phone looks at the floor, then at a table top while
walking round it at 1.2 m. On wood a 0.75 m table was found at 0.79 m; on tiles a 0.45 m table at
0.42 m; on terrazzo a 0.75 m table at 0.63 m. The scene stays within 1 px of its spot on the
table in all three. The terrazzo case shows the limit: a big table that fills the view within a
second of starting leaves almost no floor points to hold the scale, and heights come out up to
15% low. The scene still stands on the table and stays there; only its size in metres is off. In
headless Chromium with a synthetic video, the page found a 0.60 m table within a second (at
0.63 m) and put the scene on it. The probation also helps without tables: with boxes on the floor
the median anchor error fell from 1.5 to 0.7 px.

**The lens.** The page assumes the usual iPhone main camera: 26 mm equivalent, 67° across the long
side. Pro models since the 15 Pro have a 24 mm main camera, about 72°. Simulated with a 72° lens,
the scene's base stays on its spot just as well (0.48 px median). Only its size looks about 9%
off, which shows at its top (6 px). The focal length can be measured on the fly: with a wrong one
the image's turns come out bigger or smaller than the gyroscope's by the same factor. The tracker
measured 0.926 for a true 0.911. But correcting it mid-session didn't reduce the error (6.6 vs
6.2 px), because the tracked points were placed with the old one. So it isn't used; a per-model
default would be the simpler fix, if the page could tell the model (Safari doesn't say).

**Recovery after tracking is lost (floor memory).** While tracking works, every third frame adds
what the camera sees of the floor to a top-down picture of it (2 cm cells, 12 x 12 m, the running
mean of up to 15 looks per cell; `floor_map_update` in `track.ha`). Seen from above, a flat floor
looks the same whichever way the camera came from, so a view from anywhere can be compared with
it. When tracking is lost (camera covered, blank floor, a wild swing), it starts again at once,
from the last position, with the rotation from the gyroscope. Every second frame after that, the
floor it sees now is resampled from above (`floor_patch`) and searched for in the picture with
normalized cross-correlation (`floor_map_match`). The search runs first in 8 cm cells, as far as
one could have walked since the loss (0.4 m + 1.5 m/s, at most 2.5 m). Then the 5 best places are
compared again in 2 cm cells. On tiles every 30 cm looks alike in 8 cm cells, and only the fine
grain of each tile tells them apart. A winner that beats the others by 0.08 gives how far the
phone really moved. The phone's position, the tracked points and the placed scene are then
shifted back into place. The picture is only added to while the position is sure, so a wrong
position never gets into it.

Measured: the camera covered for 1 s while its holder walks on. Afterwards the scene is 0.3-1.4 px
from its spot on terrazzo, tiles, wood and carpet; without the floor memory it is 117 px off (about
280 points on screen). It also recovers after a 2 s blackout (2.4 px), a blackout during turns
(0.7 px), and 2 s on tiles while walking 1.6 m (0.6 px). While lost, a search takes up to 23 ms
every second frame.

What it doesn't do yet:
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
