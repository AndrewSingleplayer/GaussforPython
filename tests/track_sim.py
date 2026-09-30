"""Synthetic camera videos for the floor tracker (web/src/tracker.mjs + web/track.ha).

A phone's camera is simulated while its holder walks around a spot on a textured floor: the
frames are rendered by ray casting the floor, the gyroscope readings are the true rotation plus
latency, drift and noise. The tracker runs on the frames in Node with the WebAssembly module, and
its poses are compared with the truth.

    python3 tests/track_sim.py                 # run one walk, print the errors
    python3 tests/track_sim.py --gif out.gif   # and draw what the tracker follows

The key number is the anchor error: how far (in pixels) a point placed on the floor at the start
is drawn from where the real floor point is, frame after frame. That is what makes a placed
object stay put or slide.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

W, H = 202, 360                      # tracking image of a portrait 720 x 1280 camera
FOV_LONG = math.radians(67)
F = 0.5 * H / math.tan(FOV_LONG / 2)
FPS = 30


# ------------------------------------------------------------------ the floor

def resize(a, n):
    m = a.shape[0]
    x = np.linspace(0, m - 1, n)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, m - 1)
    fx = (x - i0).astype(np.float32)
    rows = a[i0] * (1 - fx)[:, None] + a[i1] * fx[:, None]
    return rows[:, i0] * (1 - fx)[None, :] + rows[:, i1] * fx[None, :]


def floor_texture(size=2048, metres=8.0, seed=1, contrast=1.0, kind="terrazzo"):
    """A floor texture. terrazzo: smooth blotches at several scales plus dots of 1-4 cm (easy).
    tiles: 30 cm tiles with faint marbling and grout lines (repetitive, mostly flat). planks: wood
    boards 15 cm wide with grain along them (few corners, stripes). carpet: fine low-contrast noise."""
    rng = np.random.default_rng(seed)
    texel = metres / size
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) * texel
    if kind == "tiles":
        tex = 0.25 * resize(rng.random((65, 65)).astype(np.float32), size)
        tex += 0.15 * resize(rng.random((257, 257)).astype(np.float32), size)
        gx = np.minimum(np.mod(xx, 0.3), 0.3 - np.mod(xx, 0.3))
        gy = np.minimum(np.mod(yy, 0.3), 0.3 - np.mod(yy, 0.3))
        tex -= 1.2 * np.clip(1.5 - np.minimum(gx, gy) / 0.004, 0, 1)
    elif kind == "planks":
        board = np.floor(xx / 0.15)
        offset = (board * 0.618 % 1.0) * 1.2
        seg = np.floor((yy + offset) / 1.2)
        shade = np.sin(board * 12.9898 + seg * 78.233) * 43758.5453 % 1.0
        grain = resize(rng.random((33, 2049)).astype(np.float32), size)
        grain = np.roll(grain, 0, axis=0)
        tex = 0.6 * shade + 0.5 * np.sin(xx * 180 + 3 * grain) * 0.3 + 0.3 * grain
        edge = np.minimum(np.mod(xx, 0.15), 0.15 - np.mod(xx, 0.15))
        tex -= 0.8 * np.clip(1.5 - edge / 0.003, 0, 1)
        ends = np.minimum(np.mod(yy + offset, 1.2), 1.2 - np.mod(yy + offset, 1.2))
        tex -= 0.8 * np.clip(1.5 - ends / 0.003, 0, 1)
    elif kind == "carpet":
        tex = resize(rng.random((1025, 1025)).astype(np.float32), size)
        tex += 0.5 * resize(rng.random((129, 129)).astype(np.float32), size)
        contrast *= 0.5
    if kind != "terrazzo":
        tex = (tex - tex.mean()) / tex.std()
        return np.clip(128 + 38 * contrast * tex, 8, 247).astype(np.float32), texel
    tex = np.zeros((size, size), np.float32)
    for cells, amp in ((6, 0.45), (24, 0.3), (96, 0.25), (384, 0.2)):
        tex += amp * resize(rng.random((cells + 1, cells + 1)).astype(np.float32), size)
    texel = metres / size
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(9000):
        r = rng.uniform(0.01, 0.04) / texel
        cx, cy = rng.uniform(0, size, 2)
        x0, x1 = int(max(0, cx - r - 1)), int(min(size, cx + r + 2))
        y0, y1 = int(max(0, cy - r - 1)), int(min(size, cy + r + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        d = np.hypot(xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy)
        a = np.clip(r + 0.5 - d, 0, 1)
        tex[y0:y1, x0:x1] = tex[y0:y1, x0:x1] * (1 - a) + a * rng.choice([0.1, 1.2])
    tex = (tex - tex.mean()) / tex.std()
    return np.clip(128 + 38 * contrast * tex, 8, 247).astype(np.float32), texel


def texture_levels(tex):
    levels = [tex]
    while levels[-1].shape[0] > 64:
        t = levels[-1]
        levels.append(0.25 * (t[0::2, 0::2] + t[1::2, 0::2] + t[0::2, 1::2] + t[1::2, 1::2]))
    return levels


def render(levels, texel, C, T, floor_z, gain=1.0, noise=2.0, rng=None, ss=2, focal=None, boxes=(), rows=None):
    """The grey image the camera at T with rotation C (camera -> world) sees, ss x ss supersampled.
    focal: the camera's real focal length (default: the one the tracker assumes). boxes: (lo, hi)
    corners of boxes standing on the floor, drawn with their own texture (points that aren't on the
    floor, which the tracker has to reject)."""
    w, h, f = W * ss, H * ss, (focal or F) * ss
    u = (np.arange(w) - (w - 1) / 2) / f
    v = (np.arange(h) - (h - 1) / 2) / f
    if rows is not None:                  # only these rows (of the final image): rolling shutter
        v = v.reshape(H, ss)[rows[0]:rows[1]].reshape(-1)
    uu, vv = np.meshgrid(u, v)
    d = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.asarray(C).T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    dz = d[..., 2]
    s = np.where(dz < -1e-3, (floor_z - T[2]) / np.minimum(dz, -1e-3), np.inf)
    hit = s < 25
    px = T[0] + s * d[..., 0]
    py = T[1] + s * d[..., 1]
    size0 = levels[0].shape[0]
    metres = size0 * texel
    out = np.full(uu.shape, 0.0, np.float32)
    # background above the floor: a plain wall, darker toward the top
    out[:] = (150 - 40 * np.clip(-vv, 0, 1))
    footprint = s / (f * texel) / np.maximum(-dz, 0.15)
    lvl = np.clip(np.floor(np.log2(np.maximum(footprint, 1e-6))), 0, len(levels) - 1).astype(int)
    for L, tex in enumerate(levels):
        m = hit & (lvl == L)
        if not m.any():
            continue
        n = tex.shape[0]
        tx = (px[m] / metres + 0.5) * n - 0.5
        ty = (py[m] / metres + 0.5) * n - 0.5
        tx = np.mod(tx, n - 1)
        ty = np.mod(ty, n - 1)
        x0 = np.floor(tx).astype(int)
        y0 = np.floor(ty).astype(int)
        ax = tx - x0
        ay = ty - y0
        x1 = np.minimum(x0 + 1, n - 1)
        y1 = np.minimum(y0 + 1, n - 1)
        out[m] = ((tex[y0, x0] * (1 - ax) + tex[y0, x1] * ax) * (1 - ay)
                  + (tex[y1, x0] * (1 - ax) + tex[y1, x1] * ax) * ay)
    for lo, hi in boxes:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - T) / d
            t2 = (hi - T) / d
        tn = np.nanmax(np.minimum(t1, t2), axis=-1)
        tf = np.nanmin(np.maximum(t1, t2), axis=-1)
        m = (tn < tf) & (tn > 0) & (tn < s)
        if m.any():
            q = T + tn[m][:, None] * d[m]
            pattern = np.sin(q[:, 0] * 41) * np.sin(q[:, 1] * 37) * np.sin(q[:, 2] * 53)
            out[m] = 128 + 90 * np.sign(pattern) * np.minimum(1, 4 * np.abs(pattern))
    nrows = H if rows is None else rows[1] - rows[0]
    img = out.reshape(nrows, ss, W, ss).mean(axis=(1, 3)) * gain
    if rng is not None and noise > 0:
        img = img + rng.normal(0, noise, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ the walk

def rot_axis(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def look_at(T, target, roll=0.0):
    """Camera -> world rotation of a camera at T looking at target (x right, y down, z forward)."""
    fwd = np.asarray(target, float) - T
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    C = np.stack([right, down, fwd], 1)
    return C @ rot_axis([0, 0, 1], roll)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Walk:
    """Stand, walk `arc` degrees around a spot on the floor keeping it in view, stand again."""

    def __init__(self, seconds=8.0, arc=150.0, radius=1.7, height=1.35, seed=2, turns=0.0, crouch=0.0, look_up=0.0):
        self.crouch = crouch                                   # metres down and up again, mid-walk
        self.look_up = look_up                                 # the spot looked at is this high (a table top)
        self.seconds, self.arc, self.radius, self.height = seconds, math.radians(arc), radius, height
        self.turns = math.radians(turns)                      # quick look away and back, peak angle
        self.anchor = np.array([0.0, radius, -height])        # the spot, north of the start
        rng = np.random.default_rng(seed)
        self.shake = rng.normal(0, 1, (6, 4))                  # hand shake: random phases/amplitudes

    def pose(self, t):
        p = smooth((t - 1.0) / (self.seconds - 2.0))
        ang = -math.pi / 2 + self.arc * p
        r = self.radius * (1 - 0.18 * math.sin(math.pi * p))
        bob = 0.015 * math.sin(2 * math.pi * 1.8 * t) * (0 < p < 1)
        T = np.array([r * math.cos(ang), self.radius + r * math.sin(ang), bob])
        if self.crouch:
            T[2] -= self.crouch * math.sin(math.pi * min(max((t - 2.0) / (self.seconds - 4.0), 0), 1)) ** 2
        T[0:2] += 0.004 * np.sin(t * np.array([7.1, 5.3]) + self.shake[0, :2])
        up = self.look_up * smooth((t - 1.0) / 1.0)            # the floor first, then up to the table top
        side = np.array([0.9, -0.5, 0.0]) * (1 - smooth((t - 1.0) / 1.0)) if self.look_up else 0.0
        look = self.anchor + side + np.array([0.12 * math.sin(0.9 * t + self.shake[1, 0]),
                                              0.10 * math.sin(0.7 * t + self.shake[1, 1]), 0.05 + up])
        C = look_at(T, look, roll=math.radians(3) * math.sin(0.5 * t + self.shake[2, 0]))
        if self.turns:                                         # two quick turns of the head/phone
            for t0 in (2.5, 5.0):
                x = (t - t0) / 0.8
                if 0 < x < 1:
                    C = rot_axis([0, 0, 1], self.turns * math.sin(math.pi * x) ** 2) @ C
        tremor = rot_axis([math.sin(3.7 * t), math.cos(4.3 * t), 0.5], math.radians(0.25) * math.sin(8 * t))
        return C @ tremor, T


def simulate(seconds=8.0, arc=150.0, latency=0.05, drift_deg_s=0.4, gyro_noise_deg=0.05,
             assumed_height=1.35, true_height=1.35, contrast=1.0, seed=3, keep_frames=False,
             true_fov=67.0, exposure_ms=0.0, boxes=False, turns=0.0, floor="terrazzo", readout_ms=0.0,
             crouch=0.0, noise=2.0, blackout=None, options=None, table=None, radius=1.7):
    rng = np.random.default_rng(seed)
    tex, texel = floor_texture(seed=seed, contrast=contrast, kind=floor)
    levels = texture_levels(tex)
    walk = Walk(seconds, arc, radius=radius, height=true_height, seed=seed, turns=turns, crouch=crouch,
                look_up=table or 0.0)
    focal = 0.5 * H / math.tan(math.radians(true_fov) / 2)
    box_list = []
    if boxes:      # a low table, a box and a chair-sized block around the walk
        fz = -true_height
        box_list = [((-1.2, 2.2, fz), (-0.6, 2.9, fz + 0.45)), ((0.9, 0.9, fz), (1.3, 1.3, fz + 0.3)),
                    ((-0.4, 3.6, fz), (0.4, 4.0, fz + 0.9))]
    if table:                     # a 0.9 x 0.8 m table under the spot looked at
        fz = -true_height
        box_list = box_list + [((-0.45, radius - 0.4, fz), (0.45, radius + 0.4, fz + table))]
    n = int(seconds * FPS)
    frames = np.zeros((n, H, W), np.uint8)
    truth = []
    # motion sensor readings at 60 Hz, each the rotation at its own time (plus drift and noise);
    # frame k shows the scene at k / FPS and reaches the page `latency` seconds later
    readings = []
    for j in range(int((seconds + latency + 0.1) * 60)):
        tj = j / 60
        Cg, _ = walk.pose(tj)
        Cg = rot_axis([0, 0, 1], math.radians(drift_deg_s) * tj) @ Cg
        Cg = Cg @ rot_axis(rng.normal(0, 1, 3), math.radians(gyro_noise_deg))
        readings.append((tj, Cg))
    gyro, times, sent = [], [], 0
    for k in range(n):
        t = k / FPS
        C, T = walk.pose(t)
        gain = 1 + 0.08 * math.sin(0.8 * t)
        if readout_ms > 0:                                 # rolling shutter: 12 bands, top first
            bands = np.linspace(0, H, 13).astype(int)
            img = np.zeros((H, W), np.float32)
            for b in range(12):
                tb = t - readout_ms / 1000 * (0.5 - (b + 0.5) / 12)
                Cs, Ts = walk.pose(tb)
                img[bands[b]:bands[b + 1]] = render(levels, texel, Cs, Ts, -true_height, gain, 0, None, 2, focal,
                                                    box_list, (bands[b], bands[b + 1]))
            frames[k] = np.clip(np.round(img + rng.normal(0, noise, img.shape)), 0, 255).astype(np.uint8)
        elif exposure_ms > 0:                              # motion blur: average over the exposure
            sub = [walk.pose(t - exposure_ms / 1000 * (j / 3)) for j in range(4)]
            img = np.mean([render(levels, texel, Cs, Ts, -true_height, gain, 0, None, 1, focal, box_list)
                           for Cs, Ts in sub], axis=0)
            frames[k] = np.clip(np.round(img + rng.normal(0, noise, img.shape)), 0, 255).astype(np.uint8)
        else:
            frames[k] = render(levels, texel, C, T, -true_height, gain, noise, rng, 2, focal, box_list)
        if blackout and blackout[0] <= t < blackout[1]:        # the camera covered: dark noise
            frames[k] = np.clip(np.round(8 + rng.normal(0, 3, (H, W))), 0, 255).astype(np.uint8)
        truth.append((C, T))
        arrive = t + latency
        new = []
        while sent < len(readings) and readings[sent][0] <= arrive:
            new.append({"t": readings[sent][0] * 1000, "C": np.asarray(readings[sent][1]).reshape(-1).tolist()})
            sent += 1
        gyro.append(new)
        times.append(arrive * 1000)
    with tempfile.TemporaryDirectory() as tmp:
        wasm = build_wasm(tmp)
        fpath, mpath, opath = (os.path.join(tmp, x) for x in ("frames.bin", "meta.json", "out.json"))
        frames.tofile(fpath)
        with open(mpath, "w") as fh:
            json.dump({"w": W, "h": H, "f": F, "n": n, "height": assumed_height, "gyro": gyro, "times": times,
                       "options": options or {}}, fh)
        subprocess.run(["node", os.path.join(HERE, "track_run.mjs"), fpath, mpath, wasm, opath], check=True)
        with open(opath) as fh:
            out = json.load(fh)

    if table:
        r = evaluate_table(truth, out, assumed_height, true_height, table, radius)
        if keep_frames:
            r["out"], r["truth"] = out, truth
        return r
    res = evaluate(truth, out, assumed_height, true_height, focal)
    if blackout:                  # how far off the spot is once the camera sees again
        after = [e for k, e in enumerate(res["anchor_errors"]) if k / FPS >= blackout[1] + 0.5]
        res["after_blackout_px"] = float(np.nanmedian(after)) if after else float("nan")
        res["relocs"] = out[-1].get("relocs", 0)
    if keep_frames:
        res["frames"], res["truth"], res["out"] = frames, truth, out
    return res


def build_wasm(tmp):
    from happ.driver import build as happ_build
    return happ_build(os.path.join(ROOT, "web", "track.ha"), ["web-wasm32"], tmp, quiet=True,
                      bridges=False)["web-wasm32"]


def project(C, T, X, f=None):
    c = np.asarray(C).T @ (np.asarray(X) - T)
    if c[2] <= 0.05:
        return None
    f = f or F
    return np.array([f * c[0] / c[2] + (W - 1) / 2, f * c[1] / c[2] + (H - 1) / 2])


def evaluate(truth, out, assumed_height, true_height, true_focal=None):
    """Anchor error: a spot placed at the start (the middle of the first frame) drawn with the
    tracker's poses (and its focal length: assumed x the scale it measured) vs. where the real spot
    is in the image (the real focal length). Positions compare after scaling by the height ratio."""
    C0, T0 = truth[0]
    d = C0 @ np.array([0, 0, 1.0])
    s = (-true_height - T0[2]) / d[2]
    anchor_true = T0 + s * d
    Ce0 = np.array(out[0]["C"]).reshape(3, 3)
    Te0 = np.array(out[0]["T"])
    de = Ce0 @ np.array([0, 0, 1.0])
    se = (-assumed_height - Te0[2]) / de[2]
    anchor_est = Te0 + se * de
    scale = assumed_height / true_height
    top = np.array([0, 0, 0.7])
    errs, pos_errs, states, ms, inliers, vecs, base = [], [], [], [], [], [], []
    for (C, T), o in zip(truth, out):
        states.append(o["state"])
        if o["C"] is None:
            o["C"] = np.eye(3).reshape(-1).tolist()
        ms.append(o["ms"])
        inliers.append(o["inliers"])
        Ce = np.array(o["C"]).reshape(3, 3)
        Te = np.array(o["T"])
        pos_errs.append(float(np.linalg.norm(Te / scale - T)))
        fe = F * o.get("fScale", 1.0)
        a = project(C, T, anchor_true, true_focal)
        b = project(Ce, Te, anchor_est, fe)
        a2 = project(C, T, anchor_true + top / scale, true_focal)
        b2 = project(Ce, Te, anchor_est + top, fe)
        if a is None or b is None:
            errs.append(float("nan"))
            vecs.append(None)
            continue
        vecs.append(b - a)
        e = float(np.linalg.norm(a - b))
        base.append(e)
        if a2 is not None and b2 is not None:
            e = max(e, float(np.linalg.norm(a2 - b2)))
        errs.append(e)
    errs = np.array(errs)
    # jitter: how much the drawn spot wobbles around the real one from one frame to the next
    steps = [np.linalg.norm(vecs[k] - vecs[k - 1]) for k in range(1, len(vecs))
             if vecs[k] is not None and vecs[k - 1] is not None]
    return {
        "jitter_px": float(np.sqrt(np.mean(np.square(steps)))) if steps else float("nan"),
        "frames": len(out),
        "anchor_px_median": float(np.nanmedian(errs)),
        "anchor_px_p95": float(np.nanpercentile(errs, 95)),
        "anchor_px_max": float(np.nanmax(errs)),
        "anchor_px_end": float(errs[-1]),
        "base_px_median": float(np.median(base)) if base else float("nan"),
        "base_px_max": float(np.max(base)) if base else float("nan"),
        "position_err_end_m": pos_errs[-1],
        "position_err_max_m": max(pos_errs),
        "path_m": float(sum(np.linalg.norm(truth[i + 1][1] - truth[i][1]) for i in range(len(truth) - 1))),
        "lost_frames": sum(s == "lost" or s == "searching" for s in states),
        "inliers_median": float(np.median(inliers)),
        "ms_median": float(np.median(ms)),
        "ms_max": float(np.max(ms)),
        "lag_ms_end": float(out[-1].get("lag", float("nan"))),
        "anchor_errors": errs.tolist(),
    }


def evaluate_table(truth, out, assumed_height, true_height, table, radius, place_at=4.0):
    """The table test: the surface found (its height above the floor), and a scene put where the
    middle of the screen meets a surface at place_at seconds: how far it is drawn from the real
    spot on the table top afterwards."""
    from importlib import import_module  # noqa: F401
    k0 = int(place_at * FPS)
    scale = assumed_height / true_height
    top = -true_height + table
    lo, hi = np.array([-0.45, radius - 0.4]), np.array([0.45, radius + 0.4])
    C, T = truth[k0]
    d = C @ np.array([0, 0, 1.0])
    s = (top - T[2]) / d[2]
    P = T + s * d
    on_table = bool(s > 0 and np.all(P[:2] >= lo) and np.all(P[:2] <= hi))
    if not on_table:
        P = T + (-true_height - T[2]) / d[2] * d
    o = out[k0]
    Ce, Te = np.array(o["C"]).reshape(3, 3), np.array(o["T"])
    de = Ce @ np.array([0, 0, 1.0])
    Pe, found = None, None
    for pl in sorted(o.get("planes") or [], key=lambda q: -q["n"]):
        se = (pl["z"] - Te[2]) / de[2]
        X = Te + se * de
        if se > 0 and inside(pl["hull"], X[0], X[1], 0.05):
            Pe, found = X, pl
            break
    if Pe is None:
        Pe = Te + (-assumed_height - Te[2]) / de[2] * de
    errs = []
    for k in range(k0, len(out)):
        a = project(truth[k][0], truth[k][1], P)
        b = project(np.array(out[k]["C"]).reshape(3, 3), np.array(out[k]["T"]), Pe)
        if a is not None and b is not None:
            errs.append(float(np.linalg.norm(a - b)))
    planes = out[-1].get("planes") or []
    main = max(planes, key=lambda q: q["n"]) if planes else None
    return {
        "planes": len(planes),
        "table_height_found": (main["z"] + assumed_height) / scale if main else float("nan"),
        "table_height_true": table,
        "true_spot_on_table": on_table,
        "placed_on_table": found is not None,
        "placed_height": float(Pe[2] + assumed_height) / scale,
        "anchor_px_median": float(np.median(errs)) if errs else float("nan"),
        "anchor_px_max": float(np.max(errs)) if errs else float("nan"),
        "lost_frames": sum(o["state"] in ("lost", "searching") for o in out),
        "ms_median": float(np.median([o["ms"] for o in out])),
        "anchor_errors": errs,
    }


def inside(hull, x, y, margin):
    if len(hull) < 3:
        return any(math.hypot(a - x, b - y) <= margin for a, b in hull)
    for i in range(len(hull)):
        (ax, ay), (bx, by) = hull[i], hull[(i + 1) % len(hull)]
        ex, ey = bx - ax, by - ay
        if (ex * (y - ay) - ey * (x - ax)) / (math.hypot(ex, ey) or 1) < -margin:
            return False
    return True


def draw_gif(res, path):
    """The frames with the anchor where the tracker puts it (red) and where it really is (green)."""
    from PIL import Image, ImageDraw
    frames, truth, out = res["frames"], res["truth"], res["out"]
    imgs = []
    C0, T0 = truth[0]
    d = C0 @ np.array([0, 0, 1.0])
    anchor = T0 + (-1.35 - T0[2]) / d[2] * d
    for k in range(0, len(frames), 2):
        im = Image.fromarray(frames[k]).convert("RGB").resize((W * 2, H * 2))
        dr = ImageDraw.Draw(im)
        C, T = truth[k]
        Ce, Te = np.array(out[k]["C"]).reshape(3, 3), np.array(out[k]["T"])
        for (CC, TT, col) in ((C, T, (40, 220, 90)), (Ce, Te, (240, 60, 40))):
            p = project(CC, TT, anchor)
            q = project(CC, TT, anchor + np.array([0, 0, 0.7]))
            if p is not None and q is not None:
                dr.line([tuple(2 * p), tuple(2 * q)], fill=col, width=3)
                dr.ellipse([2 * p[0] - 6, 2 * p[1] - 6, 2 * p[0] + 6, 2 * p[1] + 6], outline=col, width=3)
        dr.text((8, 8), f"{out[k]['state']}  {out[k]['inliers']} pts  err {res['anchor_errors'][k]:.1f}px", fill=(255, 255, 0))
        imgs.append(im)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=66, loop=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gif")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--arc", type=float, default=150.0)
    ap.add_argument("--latency", type=float, default=0.05)
    ap.add_argument("--true-height", type=float, default=1.35)
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--true-fov", type=float, default=67.0, help="the camera's real field of view (the tracker assumes 67)")
    ap.add_argument("--exposure-ms", type=float, default=0.0, help="motion blur")
    ap.add_argument("--boxes", action="store_true", help="boxes standing on the floor")
    ap.add_argument("--turns", type=float, default=0.0, help="two quick turns of this many degrees")
    ap.add_argument("--floor", default="terrazzo", choices=["terrazzo", "tiles", "planks", "carpet"])
    ap.add_argument("--readout-ms", type=float, default=0.0, help="rolling shutter")
    ap.add_argument("--crouch", type=float, default=0.0, help="go this many metres down and up again")
    ap.add_argument("--noise", type=float, default=2.0, help="camera noise, grey levels (RMS)")
    ap.add_argument("--blackout", type=float, nargs=2, help="cover the camera from .. to (seconds)")
    ap.add_argument("--no-reloc", action="store_true", help="without the floor memory")
    ap.add_argument("--table", type=float, help="a table this high (metres) under the spot looked at")
    ap.add_argument("--radius", type=float, default=1.7, help="distance kept from the spot")
    args = ap.parse_args()
    res = simulate(args.seconds, args.arc, args.latency, true_height=args.true_height,
                   contrast=args.contrast, seed=args.seed, keep_frames=bool(args.gif),
                   true_fov=args.true_fov, exposure_ms=args.exposure_ms, boxes=args.boxes, turns=args.turns,
                   floor=args.floor, readout_ms=args.readout_ms, crouch=args.crouch, noise=args.noise,
                   blackout=args.blackout, options={"relocalize": not args.no_reloc}, table=args.table,
                   radius=args.radius)
    for k, v in res.items():
        if k not in ("anchor_errors", "frames", "truth", "out"):
            print(f"{k:22s} {v:.3f}" if isinstance(v, float) else f"{k:22s} {v}")
    if args.gif:
        draw_gif(res, args.gif)
        print("wrote", args.gif)


if __name__ == "__main__":
    main()
