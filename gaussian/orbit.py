"""Orbit animation of a real splat scene, rendered by the HA++ engine, with the
measured render time of every frame written on it.

    python3 gaussian/orbit.py halo --out halo.gif
    python3 gaussian/orbit.py unicorn --frames 60 --size 640x480

Scenes: see `python3 gaussian/scenes.py`. Needs Pillow (pip install pillow) for the GIF.
The time shown is what this computer's Vulkan device took for the whole frame
(project, sort, draw). On a PC without a GPU that device is lavapipe, which
runs the GPU kernels on the CPU cores, so the numbers are far from phone speed.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import render  # noqa: E402
import scenes  # noqa: E402


def orbit_cameras(s, name, frames, width, height, elevation=0.35, fov=50.0):
    c, dist = scenes.framing(s, name)
    up = np.array(scenes.up_of(name), float)
    front = np.array([0.0, 0.0, -1.0])
    front -= up * np.dot(front, up)
    front /= np.linalg.norm(front)
    side = np.cross(up, front)
    for i in range(frames):
        a = 2 * math.pi * i / frames
        eye = c + (math.cos(a) * front + math.sin(a) * side) * 0.9 * dist + up * elevation * dist
        yield render.look_at(eye, c, width, height, fov_deg=fov, near=0.02 * dist, far=dist * 20, up=tuple(up))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", choices=list(scenes.SCENES))
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--size", default="480x360")
    ap.add_argument("--fps", type=int, default=20, help="playback speed of the GIF")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from PIL import Image, ImageDraw

    w, h = (int(v) for v in args.size.split("x"))
    s = scenes.scene(args.scene)
    pipe = render.Pipeline(render.build())
    pipe.upload(s.raw14())
    gpu = pipe.name.split(" (")[0]
    cams = list(orbit_cameras(s, args.scene, args.frames, w, h))
    pipe.render(cams[0], w, h, background=(255, 255, 255))          # warm-up (first use of the buffers)
    images, times = [], []
    for i, cam in enumerate(cams):
        t0 = time.perf_counter()
        img, pairs = pipe.render(cam, w, h, background=(255, 255, 255))
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        rgb = Image.fromarray(img.view(np.uint8).reshape(h, w, 4)[:, :, :3].copy())
        d = ImageDraw.Draw(rgb)
        lines = [f"HA++ splats: {args.scene}, {len(s):,} splats, {w}x{h}",
                 f"frame {i + 1}/{len(cams)}: {dt:.0f} ms  ({1000 / dt:.1f} fps) on {gpu}"]
        for k, text in enumerate(lines):
            x, y = 6, 6 + 14 * k
            d.rectangle([x - 3, y - 2, x + 6 * len(text) + 3, y + 11], fill=(0, 0, 0))
            d.text((x, y), text, fill=(255, 255, 255))
        images.append(rgb.convert("P", palette=Image.Palette.ADAPTIVE, colors=255))
        print(f"\rframe {i + 1}/{len(cams)}: {dt:.0f} ms, {pairs:,} tile pairs", end="", flush=True)
    out = args.out or f"{args.scene}_orbit.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=1000 // args.fps, loop=0,
                   optimize=False)
    print(f"\n{out}: {len(images)} frames, render time median {np.median(times):.0f} ms, "
          f"min {min(times):.0f} ms, max {max(times):.0f} ms on {pipe.name}")


if __name__ == "__main__":
    main()
