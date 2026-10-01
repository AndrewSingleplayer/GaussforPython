# AR in the browser: what it is, how fast, what is still needed

Live page: https://andrewsingleplayer.github.io/GaussforPython/ (tap **AR**).
Code: [`web/`](../web/README.md). How the tracking works, step by step, with measurements:
[`research/04-phones-and-webar.md`](../research/04-phones-and-webar.md).

## What it is

It is an AR engine of its own, running in a web page. It works in Safari on an iPhone. It uses
no WebXR, no ARKit, no 8th Wall and no paid SDK. It uses only what every browser gives a page:

- the camera (`getUserMedia`);
- the motion sensors (`DeviceOrientation`, `DeviceMotion`);
- WebGL2, for drawing;
- WebAssembly, for the fast parts.

At its core it works the way ARKit does. It follows points in the camera image and combines them
with the motion sensors. What differs is the inputs: ARKit gets better ones from the phone than a
web page does, and this page makes up for each one.

| Input | ARKit (native app) | This page |
|---|---|---|
| Timing between the camera and the motion sensors | exact timestamps | measured as it runs, from how the gyroscope's turning and the image's turning line up |
| Lens | calibrated at the factory | assumed: the iPhone main camera, 67° across (72° on Pro models since the 15 Pro) |
| Real-world size | camera motion, and LiDAR on Pro phones | assumed phone height above the floor: 1.35 m |
| What it maps | the room: floors, walls, and a mesh with LiDAR | the floor and horizontal surfaces above it (tables) |

**What it does:**
- **Floor and tables:** it finds the floor and tables in the camera image. A ring shows where the
  scene will go.
- **Six-way tracking:** it follows the phone's position and rotation, so you can walk around the
  scene and it stays on its spot.
- **Camera delay:** it measures how late the camera image is compared with the motion sensors
  (50-100 ms in a browser) and corrects for it.
- **Scene in step with the image:** it draws the camera image itself, so the scene and the image
  always come from the same moment. Nothing lags behind or jiggles against the image.
- **Recovery:** after tracking is lost (the camera covered, a wild swing), it recognises floor it
  has already seen and puts the scene back on its spot.
- **Light:** it matches the room's brightness and colour cast, darkens the scene's base toward
  the floor, and adds a little of the floor's colour (bounce light). These are edits made while
  drawing; the scene file is never changed.
- **Touch:** a tap places the scene, two fingers slide it, a pinch resizes it, and a twist turns
  it.

**What it isn't:**
- It isn't full SLAM. It tracks flat horizontal surfaces, not walls, and builds no 3D model of the
  room.
- It has no occlusion: a person walking in front of the scene doesn't hide it.
- Its real-world size comes from the assumed phone height and lens, not from a measurement.

**Where HA++ comes in.** These parts are written in HA++ and compiled to WebAssembly
(`web/track.ha`, `web/splatweb.ha`):
- the pixel work: corners, optical flow, the pose solve, the floor memory and the light
  measurement;
- the splat decoding and sorting.

What gets around WebXR is doing the tracking in the page, not the language: C++ or Rust compiled to
WebAssembly could do the same. What HA++ adds is that the same source also builds for a native
iPhone app (Metal) and for Android (Vulkan) ([PLATFORMS.md](PLATFORMS.md)).

**Two common objections:**
- *"iPhone has no WebXR."* That's true. Safari on iPhone has no WebXR AR mode, and Apple hasn't
  announced one. It isn't needed, because the page does its own tracking. 8th Wall worked the same
  way until its hosted service shut down in February 2026.
- *"Gaussian splats are too slow on phones."* That depends on the file and on the sorting. Here
  the files are 16 bytes per splat instead of 248 for a `.ply`. The sort runs in a background
  thread, and only when the camera moves. The quality adapts to hold the frame rate. The numbers
  are below.

## Speeds

Measured in this repository on a 4-core x86-64 virtual machine with no GPU, in Node 22 (V8, the
JavaScript engine of Chrome). An iPhone hasn't been measured yet. The page shows its own numbers
as it runs, so a screenshot on a few phones fills that gap (see *Measuring on a phone*, below).

**Splats: download, decode, sort.** The view is the one the page opens on: a portrait iPhone screen
with a 50° field of view. Decode is the best of 3 runs, sort the median of 25.

| Scene | Splats | Download | Decode, WebAssembly / JavaScript | Sort per new view, WebAssembly / JavaScript |
|---|---|---|---|---|
| Unicorn plush | 49,595 | 0.8 MB | 6.0 / 6.3 ms | 0.9 / 1.0 ms |
| Brick skull | 161,119 | 2.6 MB | 21 / 18 ms | 2.8 / 2.5 ms |
| Halo diorama | 345,002 | 5.5 MB | 49 / 43 ms | 6.2 / 5.8 ms |
| Fire pit | 561,767 | 9.0 MB | 72 / 69 ms | 10.8 / 10.6 ms |
| Horned lizard | 726,718 | 11.6 MB | 88 / 81 ms | 11.4 / 11.5 ms |
| Raccoon family | 864,532 | 13.8 MB | 98 / 107 ms | 14.0 / 12.6 ms |

- **The sort doesn't set the frame rate.** It runs in a background thread (a Web Worker), only
  after the camera moves. Drawing never waits for it: it uses the last order until the new one
  arrives. A 14 ms sort means the order is at most about one frame old.
- **WebAssembly vs JavaScript.** Once V8 has warmed up, they run at the same speed: these loops
  are simple and memory-bound. WebAssembly is faster on a page's first, small load, because it
  needs no warming up. In a fresh process the unicorn decodes in 7-8 ms in WebAssembly against
  23-34 ms in JavaScript, and the halo in 51-53 against 62-86 ms. The raccoons take 120-137 ms in
  both. Safari's JavaScript engine (JavaScriptCore) may differ; that needs a phone.
- **Download is the biggest cost on mobile data.** The raccoons are 13.8 MB here, against 214 MB as
  the original `.ply`. The scene appears only after the whole file has arrived.
- **Drawing** (the GPU's work) can't be timed on a machine without a GPU. The pixel counts that
  decide it are in [`research/02-speed.md`](../research/02-speed.md). On the phone the viewer
  holds its frame rate itself: below 42 fps it lowers the resolution first, then draws fewer of the
  faintest splats.

**Tracking: time per camera frame** (202 x 360 tracking image; WebAssembly in Node; simulated walks
from `tests/track_sim.py`):

| Walk | Median | Slowest frame |
|---|---|---|
| 150° round a spot, terrazzo floor | 1.9 ms | 9.1 ms |
| full circle, 20 s, tiles | 2.2 ms | 19.1 ms |
| two 70° turns at up to 275°/s, camera image 100 ms late | 2.2 ms | 15.7 ms |
| wood, rolling shutter, boxes on the floor | 2.3 ms | 13.7 ms |
| floor, then a table | 2.3 ms | |
| camera covered for 2 s, then found again (tiles) | 2.3 ms | 23.8 ms (a search of the floor memory) |

- A camera delivers a frame every 16-33 ms (60 or 30 fps), so tracking uses about a tenth of the
  time between frames. The slow frames are the first one, which fills the map, and the floor
  memory searches while tracking is lost.
- In headless Chromium the same tracker takes 1-5 ms per frame.
- **Accuracy:** on these walks a spot placed on the floor is drawn within 1-3 pixels of the
  tracking image (about 2.4-7 points on an iPhone screen) of where it really is. The full table is
  in [`research/04-phones-and-webar.md`](../research/04-phones-and-webar.md#floor-tracking).

**Delay.** In a browser, a camera frame arrives 50-100 ms after the motion sensor readings from
the same moment. The page draws each camera frame together with the scene posed for that same
frame, so the two never drift apart. The whole picture is about one screen frame behind Safari's own
camera preview.

### Measuring on a phone

In the viewer, the panel shows the frame rate, the time per frame, the sort time, the splats drawn,
the resolution, and which engine runs (WebAssembly or JavaScript). In AR, the line under it shows:

```
floor: <points> points, <ms per tracked frame> · camera <fps>, lag <ms> · light <gain>
```

A screenshot of these on three phones (a new iPhone, a 4-5-year-old iPhone and a mid-range
Android), with a large scene, is the missing measurement.

## What is still needed

So far the web AR has been run on one phone: an iPhone 17 Pro Max, where it works. No numbers were
recorded on it. Everything else above comes from simulations and a desktop browser.

**Before showing it to a client:**

1. **A test at the real place.** Try it on the venue's own floors and tables, with two phones. It
   takes 10 minutes. The risks:
   - **Plain or glossy floors.** The tracker needs visible texture, and polished marble shows
     reflections that move as you walk.
   - **Dim light.** It makes the camera image noisy and blurred.
   - **Glass display cases.** They reflect, and the reflections move. Placing the scene on the
     floor or an open table avoids them.
2. **More phones.** An older iPhone and a mid-range Android phone, to see the frame rate, the
   tracking time and the camera delay that each one reports.
3. **Real-world size.** The size comes from the assumed phone height (1.35 m) and lens (67°).
   On Pro iPhones (72°) the scene is about 9% too big, and a phone held lower or higher changes it
   in proportion. The scene's base still stays on its spot. Safari doesn't tell the page the phone
   model, so the options are:
   - guessing the model from the screen size;
   - a "your height" setting;
   - accepting the error, since an exhibit rarely needs exact centimetres.
4. **Start on the floor.** If a big table fills the view at the start, it can be taken for the
   floor, and heights come out up to 15% low. The scene still stands on the table and stays there.
   The hint should ask people to point at the floor first.

**For a product used by more than one museum:**

5. **Smaller and earlier downloads.** The files are already sorted most important first, so the
   scene could be shown while the rest of the file arrives. Stronger compression would help on
   mobile data.
6. **A publishing tool.** Today a developer runs `web/pack.py` and the build. A client needs
   "upload a capture, get a link and a QR code".
7. **Analytics.** How many visitors open it, for how long, and which phones fail.
8. **Wider recovery.** Recovery only works over floor the phone has already seen.

**Occlusion (later):**

| What hides the scene | How | Difficulty |
|---|---|---|
| Table edges, when the scene stands behind a table | the tables' outlines and heights are already known: draw them as invisible surfaces that hide what is behind them | easy |
| People walking in front | a person-detection model running in the browser outlines people in the camera image; their feet touch the tracked floor, which says how far away they are, so they hide the scene only when they are closer | medium; speed on older phones is the risk |
| Everything else: pillars, display cases, furniture | needs a distance for every pixel. ARKit gets it from LiDAR; in a browser it takes a depth-estimation network, which is slow on phones and blurry at edges | hard |

**Animation (to be added):**
- Playing finished 4D Gaussian splat animations, in the viewer and in AR.
- The plan, the file format, the size budget and the measured speeds are in
  [4D-SPLATS.md](4D-SPLATS.md).

**The native app (a separate path):**
- **Built but never run.** `gaussian/build_ios.py` builds `Splats.ipa` without a Mac, but the app
  has never been run on an iPhone.
- **Installing:** you install it with Sideloadly or AltStore and an Apple ID. The App Store upload
  still needs Apple's tools on macOS: a rented cloud Mac or a macOS CI runner.
- **An ARKit version:** it would get the size right and hold on blank floors (LiDAR), at the cost
  of an install.
- More in [`gaussian/README.md`](../gaussian/README.md).

## How it is checked

- **Tracking on simulated walks:** `tests/test_track.py` runs seven cases (a walk, fast turns with a
  late camera, rolling shutter, recovery, a table, the light measurement, and the wrong height and
  lens). The simulator renders a textured floor for a moving camera. Its sensors have delay, drift
  and noise.
- **The page in a real browser:** `tests/test_browser.py` opens it in headless Chromium with a fake
  camera and simulated motion sensors. It starts AR, waits for the ring, taps, and checks that the
  scene is placed with no script errors. It runs before every deploy
  (`.github/workflows/pages.yml`), so a broken page isn't published.
- **The whole suite:** 45 tests (`python3 -m unittest discover -s tests`).
