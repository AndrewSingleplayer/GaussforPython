"""Pack real 3DGS scenes into the web viewer's download format (.hspl, 16 bytes per splat).

    python3 web/pack.py                  # all scenes -> web/dist/scenes/*.hspl
    python3 web/pack.py unicorn halo     # some of them

Per scene the packer:
  - turns the scene so that its up direction is +y, and centres it on its robust centre
    (the median; captures have far-away floaters);
  - drops splats that can never be seen (opacity below 1/255) and floaters farther than
    4x the scene's framing radius;
  - orders splats by importance (opacity x area), so a viewer that must draw fewer splats
    on a slow phone just draws the first N;
  - quantizes to 16 bytes: RGBA8 colour, 3 x u16 position in the bounding box, 3 x u8 log
    scale, 3 x u8 quaternion (x, y, z; w >= 0 is implied), like Niantic's SPZ.

File layout (little endian):
   0  "HSPL"  u32 version (1)  u32 count  u32 flags
  16  f32 x3 box low corner, f32 x3 box size, f32 log-scale min, f32 log-scale range
      (the same 32 bytes as struct Box in web/splatweb.ha)
  48  f32 x3 front direction (where the default camera looks from), f32 view distance
  64  records, 16 bytes each (see web/splatweb.ha)
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gaussian"))

import scenes  # noqa: E402

SH_C0 = 0.28209479177387814
OUT = os.path.join(HERE, "dist", "scenes")


def rotation_to_y_up(up):
    """Rotation matrix taking the scene's `up` to +y (Rodrigues)."""
    a = np.asarray(up, np.float64) / np.linalg.norm(up)
    b = np.array([0.0, 1.0, 0.0])
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])   # 180 degrees about x
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1 / (1 + c))


def quat_mul(a, b):
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], 1)


def mat_to_quat(m):
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    if w > 1e-6:
        return np.array([w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    q = np.zeros(4)
    q[1 + i] = np.sqrt(max(0.0, 1 + m[i, i] - m[j, j] - m[k, k])) / 2
    q[0] = (m[k, j] - m[j, k]) / (4 * q[1 + i])
    q[1 + j] = (m[j, i] + m[i, j]) / (4 * q[1 + i])
    q[1 + k] = (m[k, i] + m[i, k]) / (4 * q[1 + i])
    return q


def pack(name, out_dir=OUT, max_splats=None):
    s = scenes.scene(name)
    center, dist = scenes.framing(s, name)
    data = pack_scene(s, center, dist, scenes.up_of(name), max_splats)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.hspl")
    with open(path, "wb") as f:
        f.write(data)
    return path, len(s), int.from_bytes(data[8:12], "little")


def pack_scene(s, center, dist, up, max_splats=None):
    """The .hspl bytes for a scenes.Scene, framed around `center` at `dist`, with `up` turned to +y."""
    R = rotation_to_y_up(up)
    pos = (s.pos.astype(np.float64) - center) @ R.T
    alpha = 1 / (1 + np.exp(-s.opacity_logit.astype(np.float64)))
    radius = dist / 2
    keep = (alpha >= 1 / 255) & (np.linalg.norm(pos, axis=1) < 4 * radius)
    rq = np.tile(mat_to_quat(R), (len(pos), 1))
    rot = s.rot.astype(np.float64)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    rot = quat_mul(rq, rot)                         # rotate every splat's orientation too
    log_scale = s.log_scale.astype(np.float64)
    scale = np.exp(log_scale)
    importance = alpha * (scale[:, 0] * scale[:, 1] + scale[:, 1] * scale[:, 2] + scale[:, 0] * scale[:, 2])
    idx = np.nonzero(keep)[0]
    idx = idx[np.argsort(-importance[idx], kind="stable")]
    if max_splats:
        idx = idx[:max_splats]
    pos, rot, log_scale, alpha = pos[idx], rot[idx], log_scale[idx], alpha[idx]
    rgb = np.clip(0.5 + SH_C0 * s.sh[idx, 0, :].astype(np.float64), 0, 1)

    lo, hi = pos.min(0), pos.max(0)
    size = np.maximum(hi - lo, 1e-9)
    lmin, lmax = log_scale.min(), log_scale.max()
    lrange = max(lmax - lmin, 1e-6)
    rec = np.zeros(len(idx), np.dtype([("rgba", "u1", 4), ("pos", "<u2", 3), ("scale", "u1", 3),
                                        ("rot", "u1", 3)]))
    rec["rgba"][:, :3] = np.round(rgb * 255)
    rec["rgba"][:, 3] = np.round(alpha * 255)
    rec["pos"] = np.round((pos - lo) / size * 65535)
    rec["scale"] = np.round((log_scale - lmin) / lrange * 255)
    rot = np.where(rot[:, :1] < 0, -rot, rot)       # w >= 0
    rec["rot"] = np.clip(np.round((rot[:, 1:] + 1) * 127.5), 0, 255)

    front = R @ np.array([0.0, 0.0, -1.0])          # gaussian/orbit.py views the captures from -z
    header = struct.pack("<4sIII3f3fff3ff", b"HSPL", 1, len(idx), 1, *lo, *size, lmin, lrange, *front,
                         dist * 0.9)
    header += b"\0" * (64 - len(header))
    return header + rec.tobytes()


def main():
    names = sys.argv[1:] or list(scenes.SCENES)
    for name in names:
        path, before, after = pack(name)
        print(f"{name:8s} {before:>8,} -> {after:>8,} splats  {os.path.getsize(path) / 1e6:6.1f} MB  {path}")


if __name__ == "__main__":
    main()
