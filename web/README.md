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

## Files

| File | What |
|---|---|
| `splatweb.ha` | decode + cull + sort, in HA++ (compiled with `happ build -t web`) |
| `pack.py` | real scenes (`gaussian/scenes.py`) → `.hspl` download format |
| `build.py` | builds everything into `dist/`: the page as a full document (`index.html`) and as a fragment for a claude.ai artifact (`app.html`) |
| `src/viewer.js` | WebGL2 drawing, touch controls, quality control |
| `src/worker.js`, `src/engine.mjs` | the worker and the engine (HA++ WebAssembly, or the JavaScript fallback) |

The scene files in `dist/scenes/` are WebAssembly modules whose only content is one data segment holding the `.hspl` bytes, at the offset listed in `scenes.json`. Some hosts serve only web file types; a claude.ai artifact is one. The viewer reads the bytes directly and never needs to run these modules.

Tests: `tests/test_web.py` runs the pack → decode → sort path in Node, through both the WebAssembly engine and the JavaScript one, and checks it against NumPy.

Scenes: [Babylon.js Assets](https://github.com/BabylonJS/Assets), CC BY 4.0.
