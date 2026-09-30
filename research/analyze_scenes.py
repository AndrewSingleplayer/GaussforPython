"""Measurements on real 3DGS captures that decide how the renderers are built.

    python3 research/analyze_scenes.py            # all scenes, phone-sized view
    python3 research/analyze_scenes.py halo unicorn

For each scene, from one camera that frames it (the same framing as gaussian/orbit.py),
at an iPhone-like render size (portrait 590x1280, i.e. half of a 1179x2556 screen):

  invisible      splats whose opacity is below 1/255: they can never change a pixel
  3-sigma box    screen pixels covered by the v1 bounds: a square of side 2*ceil(3*sqrt(lambda_max))
  tight quad     pixels of an oriented rectangle along the ellipse axes, cut where the Gaussian
                 falls below 1/255 of the splat's opacity: extent sqrt(2 ln(255 a)) sigma per axis
  tile pairs     (splat, 16x16 tile) pairs the v1 compute rasterizer sorts, with the 3-sigma
                 square vs with exact ellipse-tile overlap at the opacity cut
  fp16 cov       splats whose 3D covariance loses precision as f16 (entries below the smallest
                 normal f16, 6.1e-5): v1 stores the covariance as six f16 values
  SH energy      share of colour variation that is view-dependent (degree 1-3), for scenes with SH
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gaussian"))

import scenes  # noqa: E402

W, H = 590, 1280
TILE = 16


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def quat_to_mat(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def camera(s, name):
    c, dist = scenes.framing(s, name)
    up = np.array(scenes.up_of(name), float)
    front = np.array([0.0, 0.0, -1.0]) - up * np.dot([0.0, 0.0, -1.0], up)
    front /= np.linalg.norm(front)
    eye = c + front * 0.9 * dist + up * 0.35 * dist
    fwd = (c - eye) / np.linalg.norm(c - eye)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    f = 0.5 * H / np.tan(np.radians(50) / 2)        # 50 degree vertical field of view
    return R, -R @ eye, f, 0.02 * dist, 20 * dist


def analyze(name):
    s = scenes.scene(name)
    n = len(s)
    alpha = sigmoid(s.opacity_logit.astype(np.float64))
    out = {"scene": name, "splats": n, "invisible": float(np.mean(alpha < 1 / 255))}

    # 3D covariance, as pack_splats computes it
    Rm = quat_to_mat(s.rot.astype(np.float64))
    sc = np.exp(s.log_scale.astype(np.float64))
    M = Rm * sc[:, None, :]
    cov3 = M @ np.transpose(M, (0, 2, 1))
    tri = np.stack([cov3[:, 0, 0], cov3[:, 0, 1], cov3[:, 0, 2], cov3[:, 1, 1], cov3[:, 1, 2], cov3[:, 2, 2]], 1)
    nonzero = np.abs(tri) > 0
    tiny = (np.abs(tri) < 6.1e-5) & nonzero
    diag_tiny = (np.stack([cov3[:, 0, 0], cov3[:, 1, 1], cov3[:, 2, 2]], 1) < 6.1e-5).any(1)
    out["fp16_offdiag_lost"] = float(np.mean(tiny.any(1)))
    out["fp16_diag_lost"] = float(np.mean(diag_tiny))

    # project (EWA, as in splat.ha)
    R, t, f, near, far = camera(s, name)
    pc = s.pos.astype(np.float64) @ R.T + t
    vis = (pc[:, 2] > near) & (pc[:, 2] < far) & (alpha >= 1 / 255)
    pc, cov3, alpha = pc[vis], cov3[vis], alpha[vis]
    z = pc[:, 2]
    lim = 1.3 * np.array([W, H]) * 0.5 / f
    tx = np.clip(pc[:, 0] / z, -lim[0], lim[0]) * z
    ty = np.clip(pc[:, 1] / z, -lim[1], lim[1]) * z
    J = np.zeros((len(z), 2, 3))
    J[:, 0, 0] = f / z
    J[:, 0, 2] = -f * tx / (z * z)
    J[:, 1, 1] = f / z
    J[:, 1, 2] = -f * ty / (z * z)
    Wm = np.broadcast_to(R, (len(z), 3, 3))
    T = J @ Wm
    cov2 = T @ cov3 @ np.transpose(T, (0, 2, 1))
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = a * c - b * b
    mid = 0.5 * (a + c)
    disc = np.sqrt(np.maximum(0.1, mid * mid - det))
    l1, l2 = mid + disc, np.maximum(mid - disc, 1e-12)
    cx = f * pc[:, 0] / z + W / 2
    cy = f * pc[:, 1] / z + H / 2

    # v1 bounds: square of radius ceil(3 sqrt(l1)), clipped to the screen
    r3 = np.ceil(3 * np.sqrt(l1))
    on = (cx + r3 > 0) & (cx - r3 < W) & (cy + r3 > 0) & (cy - r3 < H)
    out["on_screen"] = int(on.sum())

    def clipped_area(x0, x1, y0, y1):
        return (np.clip(x1, 0, W) - np.clip(x0, 0, W)) * (np.clip(y1, 0, H) - np.clip(y0, 0, H))

    box_px = clipped_area(cx - r3, cx + r3, cy - r3, cy + r3)[on].sum()
    # tight oriented quad: half extents k*sqrt(l1), k*sqrt(l2), k = sqrt(2 ln(255 a))
    k = np.sqrt(2 * np.log(np.maximum(255 * alpha, 1.0)))
    quad_px = (4 * k * np.sqrt(l1) * k * np.sqrt(l2))[on].sum()   # before screen clipping (upper bound)
    out["px_box"] = float(box_px)
    out["px_quad"] = float(quad_px)
    out["overdraw_box"] = float(box_px / (W * H))
    out["overdraw_quad"] = float(quad_px / (W * H))

    # tile pairs: v1 square vs exact ellipse/tile overlap at the opacity cut
    tiles_x, tiles_y = -(-W // TILE), -(-H // TILE)
    lo_x = np.clip(np.floor((cx - r3) / TILE), 0, tiles_x)
    hi_x = np.clip(np.floor((cx + r3 + 15) / TILE), 0, tiles_x)
    lo_y = np.clip(np.floor((cy - r3) / TILE), 0, tiles_y)
    hi_y = np.clip(np.floor((cy + r3 + 15) / TILE), 0, tiles_y)
    pairs_v1 = ((hi_x - lo_x) * (hi_y - lo_y))[on].sum()
    # ellipse test per tile: the tile touches the ellipse {d : d^T S^-1 d <= k^2} iff the closest point of
    # the tile rectangle to the center (in the ellipse's metric) is inside; evaluated exactly for small splats
    # and bounded by the tight bounding box of the ellipse for the rest (the box is exact for the extent)
    ex = k * np.sqrt(a)          # half width of the ellipse's axis-aligned bounding box at the opacity cut
    ey = k * np.sqrt(c)
    lo_x2 = np.clip(np.floor((cx - ex) / TILE), 0, tiles_x)
    hi_x2 = np.clip(np.floor((cx + ex) / TILE) + 1, 0, tiles_x)
    lo_y2 = np.clip(np.floor((cy - ey) / TILE), 0, tiles_y)
    hi_y2 = np.clip(np.floor((cy + ey) / TILE) + 1, 0, tiles_y)
    pairs_bbox = ((hi_x2 - lo_x2) * (hi_y2 - lo_y2))[on].sum()
    # exact count for splats spanning few tiles, bbox count for the (rare) huge ones
    ia, ib, ic = c / det, -b / det, a / det      # conic
    exact = 0
    idx = np.nonzero(on & ((hi_x2 - lo_x2) * (hi_y2 - lo_y2) <= 64))[0]
    big = np.nonzero(on & ((hi_x2 - lo_x2) * (hi_y2 - lo_y2) > 64))[0]
    exact += ((hi_x2 - lo_x2) * (hi_y2 - lo_y2))[big].sum()
    for ty0 in range(8):
        for tx0 in range(8):
            gx = lo_x2[idx] + tx0
            gy = lo_y2[idx] + ty0
            inside = (gx < hi_x2[idx]) & (gy < hi_y2[idx])
            # closest point of the tile to the center, then the ellipse test there; for an ellipse the true
            # minimum over the rectangle can be lower than at the closest point: take the minimum over the
            # closest point and the tile's 4 edges' minima (quadratic along each edge) to be exact
            x0, x1 = gx * TILE, gx * TILE + TILE
            y0, y1 = gy * TILE, gy * TILE + TILE
            dx0, dx1 = x0 - cx[idx], x1 - cx[idx]
            dy0, dy1 = y0 - cy[idx], y1 - cy[idx]
            A, B, C = ia[idx], ib[idx], ic[idx]
            q = lambda X, Y: A * X * X + 2 * B * X * Y + C * Y * Y
            best = np.full(len(idx), np.inf)
            inside_c = (dx0 <= 0) & (dx1 >= 0) & (dy0 <= 0) & (dy1 >= 0)
            best = np.where(inside_c, 0.0, best)
            for X in (dx0, dx1):              # vertical edges: minimize over Y in [dy0, dy1]
                Y = np.clip(-B * X / C, dy0, dy1)
                best = np.minimum(best, q(X, Y))
            for Y in (dy0, dy1):              # horizontal edges
                X = np.clip(-B * Y / A, dx0, dx1)
                best = np.minimum(best, q(X, Y))
            exact += np.sum(inside & (best <= k[idx] ** 2))
    out["pairs_v1"] = int(pairs_v1)
    out["pairs_bbox"] = int(pairs_bbox)
    out["pairs_exact"] = int(exact)

    # view-dependent colour energy
    if s.sh.shape[1] > 1:
        dc = s.sh[:, 0, :].astype(np.float64)
        rest = s.sh[:, 1:, :].astype(np.float64)
        # colour variation over directions: DC varies across splats; the SH part varies with direction.
        # energy of the view-dependent part relative to the DC part (orthonormal basis: sum of squares)
        out["sh_rest_energy"] = float((rest ** 2).sum() / ((dc - dc.mean(0)) ** 2).sum())
        # 1-3 degree energy split
        e1 = (s.sh[:, 1:4] ** 2).sum()
        e2 = (s.sh[:, 4:9] ** 2).sum() if s.sh.shape[1] > 4 else 0
        e3 = (s.sh[:, 9:16] ** 2).sum() if s.sh.shape[1] > 9 else 0
        tot = e1 + e2 + e3
        out["sh_split"] = (float(e1 / tot), float(e2 / tot), float(e3 / tot))
    return out


def main():
    names = sys.argv[1:] or list(scenes.SCENES)
    rows = [analyze(n) for n in names]
    print(f"Render size {W}x{H} (half of an iPhone 15 screen), 50 degree vertical FOV, framing camera\n")
    print("| scene | splats | invisible | on screen | overdraw 3-sigma box | overdraw tight quad | "
          "tile pairs v1 | tile pairs bbox | tile pairs exact | fp16 cov loses entries |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['scene']} | {r['splats']:,} | {r['invisible']:.1%} | {r['on_screen']:,} | "
              f"{r['overdraw_box']:.0f}x | {r['overdraw_quad']:.0f}x | {r['pairs_v1']:,} | {r['pairs_bbox']:,} | "
              f"{r['pairs_exact']:,} | {r['fp16_offdiag_lost']:.1%} (diag {r['fp16_diag_lost']:.1%}) |")
    print()
    for r in rows:
        if "sh_rest_energy" in r:
            e1, e2, e3 = r["sh_split"]
            print(f"{r['scene']}: view-dependent SH energy = {r['sh_rest_energy']:.1%} of the DC colour "
                  f"variation; split degree 1/2/3 = {e1:.0%}/{e2:.0%}/{e3:.0%}")


if __name__ == "__main__":
    main()
