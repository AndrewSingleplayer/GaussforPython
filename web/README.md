# HA++ Splats: the web viewer

A Gaussian splat viewer that runs in a web page, including Safari on an iPhone. It shows six real 3DGS captures.

```sh
python3 web/build.py                        # -> web/dist/ (downloads and packs the scenes once)
python3 -m http.server -d web/dist 8000     # then open http://localhost:8000
```

## How it is fast on a phone

| Step | Where | What |
|---|---|---|
| Download | once | 16 bytes per splat (`web/pack.py`), most important splats first. Floaters and invisible splats are removed. |
| Decode | Web Worker, **HA++ → WebAssembly** | `decode_splats` in `splatweb.ha` rebuilds each splat's 3D covariance from scale + rotation. It stores the covariance as f16 divided by the splat's largest variance: raw f16 would lose the small splats (about 90% of them in real captures). |
| Cull + sort | Web Worker, **HA++ → WebAssembly** | `sort_splats` in `splatweb.ha` drops splats outside the view, then orders the rest back to front with a 16-bit counting sort. It runs off the main thread, so drawing never waits for it. |
| Draw | GPU, WebGL2 | One quad per splat, along the axes of its projected ellipse (EWA, as in 3DGS). The quad is cut where the splat's contribution falls below 1/255, which is 2–5× fewer pixels than a 3-sigma square on the real scenes. The GPU's blending composites back to front. |
| Adapt | main thread | **Auto** lowers the resolution first, then draws fewer of the least important splats, until the frame rate holds. Nothing is redrawn while nothing moves. |

If a browser refuses WebAssembly, the same decode and sort run in JavaScript (`src/engine.mjs`). The page shows which one is running.

## AR without WebXR

Safari on iPhone has no WebXR, so the page does what ARKit would: it finds the floor and follows it.

| Step | Where | What |
|---|---|---|
| Rotation | main thread | the motion sensors (`DeviceOrientation`) give which way the camera points and which way is down |
| Floor | main thread | the horizontal plane 1.35 m below the phone (a phone held by a standing adult). A ring shows where the middle of the screen meets it; a tap stands the scene there, 0.7 m tall, with a soft shadow |
| Floor tracking | Web Worker, **HA++ → WebAssembly** | `track.ha`: corners on the floor (Shi-Tomasi), followed from frame to frame (pyramidal Lucas-Kanade, checked backward), the camera pose that puts them where they are seen (robust Gauss-Newton). `src/tracker.mjs` keeps the map of floor points. Every point lies on the floor, so its depth is known at once and the floor's height sets the scale |
| Tables | Web Worker | points that don't fit the floor are triangulated as the phone moves; 8 or more at one height make a surface (a table); the ring and the scene go on it, and found surfaces are outlined before placing |
| Recovery | Web Worker, **HA++ → WebAssembly** | a top-down picture of the floor (`floor_map_*` in `track.ha`); after tracking is lost, the floor seen now is found in it again and everything is put back in place |
| Light | Web Worker + GPU | `light_stats` in `track.ha` measures the room's brightness and colour cast and the floor's colour under the scene; the splats are drawn to match (brightness, tint, contact shading, floor bounce), edited in the shader as they are drawn, never in the data. "Light: original" turns it off |
| Timing | Web Worker | camera frames reach the page 50-100 ms after the motion sensor readings. The tracker measures that delay from how the gyroscope's turning and the image's turning line up, and a search of the whole image's shift catches what the gyroscope gets wrong |

On simulated walks (`tests/track_sim.py`: a textured floor rendered for a moving camera, sensors with delay, drift and noise) a spot on the floor is drawn within 1-3 pixels of the tracking image of where it really is. That holds on terrazzo, tiles, wood and carpet, through 275°/s turns, rolling shutter and motion blur, and after a full 360° walk around it. Each frame takes about 2 ms. `tests/test_track.py` runs four of these walks.

## Files

| File | What |
|---|---|
| `splatweb.ha` | decode + cull + sort, in HA++ (compiled with `happ build -t web`) |
| `pack.py` | real scenes (`gaussian/scenes.py`) → `.hspl` download format |
| `build.py` | builds everything into `dist/`: the page as a full document (`index.html`) and as a fragment for a claude.ai artifact (`app.html`) |
| `src/viewer.js` | WebGL2 drawing, touch controls, quality control |
| `src/worker.js`, `src/engine.mjs` | the worker and the engine (HA++ WebAssembly, or the JavaScript fallback) |
| `track.ha`, `src/tracker.mjs`, `src/track-worker.js` | floor tracking for AR: the pixel work in HA++, the map and the timing in JavaScript |
| `src/ar.js` | camera, motion sensors, the floor, placing the scene |

The scene files in `dist/scenes/` are WebAssembly modules whose only content is one data segment holding the `.hspl` bytes, at the offset listed in `scenes.json`. Some hosts serve only web file types; a claude.ai artifact is one. The viewer reads the bytes directly and never needs to run these modules.

Tests: `tests/test_web.py` runs the pack → decode → sort path in Node, through both the WebAssembly engine and the JavaScript one, and checks it against NumPy.

Scenes: [Babylon.js Assets](https://github.com/BabylonJS/Assets), CC BY 4.0.
