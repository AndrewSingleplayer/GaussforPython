"""Independent NumPy implementation of gaussian/splat.ha v2 (for testing the GPU pipeline)."""

import numpy as np

TILE = 16
SH_C0 = np.float32(0.28209479177387814)
F = np.float32


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def quat_to_mat3(q):
    n = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = n[:, 0], n[:, 1], n[:, 2], n[:, 3]
    cols = [np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], 1),
            np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], 1),
            np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)]
    return np.stack(cols, 2)          # (n, 3 rows, 3 cols): column-major like HA++


def pack_unorm4(c):
    u = (np.clip(c, 0, 1) * F(255) + F(0.5)).astype(np.uint32)
    return u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16) | (u[:, 3] << 24)


def unpack_unorm4(u):
    return np.stack([(u >> s) & 255 for s in (0, 8, 16, 24)], -1).astype(np.float32) * F(0.00392156862745098)


def half_pair(a, b):
    lo = a.astype(np.float16).view(np.uint16).astype(np.uint32)
    hi = b.astype(np.float16).view(np.uint16).astype(np.uint32)
    return lo | (hi << 16)


def unpack_half2(u):
    lo = (u & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (u >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
    return lo, hi


def pack(raw):
    raw = raw.astype(np.float32)
    r = quat_to_mat3(raw[:, 6:10])
    s = np.exp(raw[:, 3:6])
    m = r * s[:, None, :]
    full = m @ np.transpose(m, (0, 2, 1))
    scale = np.maximum(np.maximum(full[:, 0, 0], full[:, 1, 1]), np.maximum(full[:, 2, 2], F(1e-30)))
    cov = full * (F(1) / scale)[:, None, None]
    rgb = np.clip(raw[:, 11:14] * SH_C0 + F(0.5), 0, 1)
    color = pack_unorm4(np.concatenate([rgb, sigmoid(raw[:, 10:11])], 1))
    out = np.zeros((len(raw), 8), np.uint32)
    out[:, :3] = raw[:, :3].view(np.uint32)
    out[:, 3] = color
    out[:, 4] = half_pair(cov[:, 0, 0], cov[:, 1, 0])
    out[:, 5] = half_pair(cov[:, 2, 0], cov[:, 1, 1])
    out[:, 6] = half_pair(cov[:, 2, 1], cov[:, 2, 2])
    out[:, 7] = scale.astype(np.float32).view(np.uint32)
    return out


def preprocess(packed, cam, width, height):
    tiles_x, tiles_y = (width + TILE - 1) // TILE, (height + TILE - 1) // TILE
    view = np.asarray(cam["view"], np.float32)
    pos = packed[:, :3].view(np.float32)
    pc = pos @ view[:3, :3].T + view[:3, 3]
    ok = (pc[:, 2] >= F(cam["near"])) & (pc[:, 2] <= F(cam["far"]))
    xx, xy = unpack_half2(packed[:, 4])
    xz, yy = unpack_half2(packed[:, 5])
    yz, zz = unpack_half2(packed[:, 6])
    sigma = np.stack([np.stack([xx, xy, xz], 1), np.stack([xy, yy, yz], 1), np.stack([xz, yz, zz], 1)], 1)
    sigma = sigma * packed[:, 7].view(np.float32)[:, None, None]
    w = view[:3, :3]
    m = w @ sigma @ w.T
    focal = np.array([cam["fx"], cam["fy"]], np.float32)
    lim = F(1.3) * np.array([width, height], np.float32) * F(0.5) / focal
    z = pc[:, 2]
    tx = np.clip(pc[:, 0] / z, -lim[0], lim[0]) * z
    ty = np.clip(pc[:, 1] / z, -lim[1], lim[1]) * z
    z2 = z * z
    zero = np.zeros_like(z)
    j0 = np.stack([focal[0] / z, zero, -focal[0] * tx / z2], 1)
    j1 = np.stack([zero, focal[1] / z, -focal[1] * ty / z2], 1)
    mj0 = np.einsum("nij,nj->ni", m, j0)
    mj1 = np.einsum("nij,nj->ni", m, j1)
    a = (j0 * mj0).sum(1) + F(0.3)
    b = (j0 * mj1).sum(1)
    c = (j1 * mj1).sum(1) + F(0.3)
    det = a * c - b * b
    ok &= det > 0
    # v2: the bounding box of the ellipse where opacity * exp(-q/2) >= 1/255
    opacity = ((packed[:, 3] >> 24).astype(np.float32)) * F(0.00392156862745098)
    cut = F(2) * np.log(np.maximum(F(255) * opacity, F(1))).astype(np.float32)
    ok &= cut > 0
    with np.errstate(invalid="ignore"):
        half = np.sqrt(cut[:, None] * np.stack([a, c], 1))
    center = np.stack([focal[0] * pc[:, 0] / z + F(cam["cx"]), focal[1] * pc[:, 1] / z + F(cam["cy"])], 1)
    tiles = np.array([tiles_x, tiles_y], np.float32)
    with np.errstate(invalid="ignore"):
        lo = np.clip(np.floor((center - half) / F(16)), 0, tiles).astype(np.uint32)
        hi = np.clip(np.floor((center + half) / F(16)) + F(1), 0, tiles).astype(np.uint32)
    count = (hi[:, 0] - lo[:, 0]) * (hi[:, 1] - lo[:, 1])
    ok &= count > 0
    with np.errstate(all="ignore"):
        conic = np.stack([c, -b, a], 1) / det[:, None]
        dz = np.clip((z - F(cam["near"])) / F(cam["far"] - cam["near"]), 0, 1)
    depth = (dz * F(65535)).astype(np.uint32)
    return dict(ok=ok, center=center, conic=conic, color=packed[:, 3], lo=lo, hi=hi, depth=depth,
                tiles_x=tiles_x, tiles_y=tiles_y)


def sorted_pairs(pp):
    keys, vals = [], []
    for i in np.nonzero(pp["ok"])[0]:
        for y in range(pp["lo"][i, 1], pp["hi"][i, 1]):
            for x in range(pp["lo"][i, 0], pp["hi"][i, 0]):
                keys.append(((y * pp["tiles_x"] + x) << 16) | int(pp["depth"][i]))
                vals.append(i)
    keys = np.array(keys, np.uint32)
    vals = np.array(vals, np.uint32)
    order = np.argsort(keys, kind="stable")
    return keys[order], vals[order]


def render(packed, cam, width, height, background=(0, 0, 0)):
    pp = preprocess(packed, cam, width, height)
    keys, vals = sorted_pairs(pp)
    img = np.zeros((height, width), np.uint32)
    bg = np.array(background, np.float32) * F(0.00392156862745098)
    tiles = keys >> 16
    col = unpack_unorm4(pp["color"])
    for ty in range(pp["tiles_y"]):
        for tx in range(pp["tiles_x"]):
            tile = ty * pp["tiles_x"] + tx
            lst = vals[tiles == tile]
            ys, xs = np.mgrid[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE]
            px, py = xs.astype(np.float32).ravel(), ys.astype(np.float32).ravel()
            trans = np.ones_like(px)
            rgb = np.zeros((len(px), 3), np.float32)
            done = np.zeros(len(px), bool)
            for s in lst:
                dx = pp["center"][s, 0] - px
                dy = pp["center"][s, 1] - py
                a, b, c = pp["conic"][s]
                power = F(-0.5) * (a * dx * dx + c * dy * dy) - b * dx * dy
                alpha = np.minimum(F(0.99), col[s, 3] * np.exp(power))
                use = (~done) & (power <= 0) & (alpha >= F(0.00392156862745098))
                nxt = trans * (F(1) - alpha)
                stop = use & (nxt < F(0.0001))
                done |= stop
                add = use & ~stop
                rgb[add] += col[s, :3] * (alpha[add] * trans[add])[:, None]
                trans = np.where(add, nxt, trans)
            final = rgb + bg * trans[:, None]
            final = np.concatenate([final, np.ones((len(px), 1), np.float32)], 1)
            packed_px = pack_unorm4(final).reshape(TILE, TILE)
            h = min(TILE, height - ty * TILE)
            w = min(TILE, width - tx * TILE)
            img[ty * TILE:ty * TILE + h, tx * TILE:tx * TILE + w] = packed_px[:h, :w]
    return img, keys, vals
