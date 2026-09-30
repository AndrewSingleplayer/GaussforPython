// Floor tracking for the web AR mode, compiled to WebAssembly (`happ build web/track.ha -t web`).
// It runs in a Web Worker on the camera frames; web/src/tracker.mjs decides what to call when.
//
//   gray_image    RGBA camera frame -> grey levels
//   pyramid       half-size copies of an image, to follow large motions
//   corners       points worth following (Shi-Tomasi): the best one in each free cell of a grid
//   track_points  pyramidal Lucas-Kanade optical flow: where points of one frame went in the next
//   solve_pose    the camera's rotation and position from points on the floor whose place is known
//   light_stats   the light in the frame (brightness, colour), to draw the scene to match it
//
// Images are 8-bit grey, row by row, with pixel centres at integer coordinates. A pyramid is its
// levels one after another: level l is (w >> l) x (h >> l) pixels, each the mean of 2x2 pixels
// of the level below.

// ------------------------------------------------------------------ images

export fn gray_image(rgba: *u8, n: u32, out: *u8) {
    for i in 0..n {
        let p = rgba + i * 4;
        out[i] = ((p[0] as u32 * 77 + p[1] as u32 * 150 + p[2] as u32 * 29 + 128) >> 8) as u8;
    }
}

export fn level_offset(w: u32, h: u32, l: u32) -> u32 {
    var o: u32 = 0;
    for k in 0..l { o += (w >> k) * (h >> k); }
    return o;
}

// Fills levels 1 .. levels-1 from level 0.
export fn pyramid(img: *u8, w: u32, h: u32, levels: u32) {
    for l in 1..levels {
        let src = img + level_offset(w, h, l - 1);
        let dst = img + level_offset(w, h, l);
        let sw = w >> (l - 1);
        let dw = w >> l;
        let dh = h >> l;
        for y in 0..dh {
            let r0 = src + 2 * y * sw;
            let r1 = r0 + sw;
            let d = dst + y * dw;
            for x in 0..dw {
                let s = r0[2 * x] as u32 + r0[2 * x + 1] as u32 + r1[2 * x] as u32 + r1[2 * x + 1] as u32;
                d[x] = ((s + 2) >> 2) as u8;
            }
        }
    }
}

// size x size samples starting at (x0, y0), bilinear. All samples share the same fractions.
// The caller keeps the window inside the image.
fn window(img: *u8, w: u32, x0: f32, y0: f32, size: u32, out: *f32) {
    let fx = floor(x0);
    let fy = floor(y0);
    let ax = x0 - fx;
    let ay = y0 - fy;
    let w00 = (1.0 - ax) * (1.0 - ay);
    let w10 = ax * (1.0 - ay);
    let w01 = (1.0 - ax) * ay;
    let w11 = ax * ay;
    let base = img + (fy as u32) * w + (fx as u32);
    for j in 0..size {
        let r0 = base + j * w;
        let r1 = r0 + w;
        let o = out + j * size;
        for i in 0..size {
            o[i] = w00 * r0[i] as f32 + w10 * r0[i + 1] as f32 + w01 * r1[i] as f32 + w11 * r1[i + 1] as f32;
        }
    }
}

// ------------------------------------------------------------------ corners

// 5-pixel box sum along x, then along y, of the interior of `a` (rows and columns 3 .. n-4 are
// valid afterwards). tmp has w * h entries.
fn box5(a: *f32, tmp: *f32, w: u32, h: u32) {
    for y in 1..h - 1 {
        for x in 3..w - 3 {
            let i = y * w + x;
            tmp[i] = a[i - 2] + a[i - 1] + a[i] + a[i + 1] + a[i + 2];
        }
    }
    for y in 3..h - 3 {
        for x in 3..w - 3 {
            let i = y * w + x;
            a[i] = tmp[i - 2 * w] + tmp[i - w] + tmp[i] + tmp[i + w] + tmp[i + 2 * w];
        }
    }
}

// The best corner in each cell of a `cell` x `cell` grid whose skip byte is 0, at least `border`
// pixels from the edges. Score: the smaller eigenvalue of the gradient matrix over 5x5 pixels
// (Shi-Tomasi), as a mean squared gradient. Writes (x, y, score) per corner; returns how many.
// sxx, sxy, syy and tmp have w * h entries each.
export fn corners(img: *u8, w: u32, h: u32, cell: u32, skip: *u8, border: u32,
                  sxx: *f32, sxy: *f32, syy: *f32, tmp: *f32, out: *f32) -> u32 {
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = 0.5 * (img[i + 1] as f32 - img[i - 1] as f32);
            let gy = 0.5 * (img[i + w] as f32 - img[i - w] as f32);
            sxx[i] = gx * gx;
            sxy[i] = gx * gy;
            syy[i] = gy * gy;
        }
    }
    box5(sxx, tmp, w, h);
    box5(sxy, tmp, w, h);
    box5(syy, tmp, w, h);
    let cw = w / cell;
    let ch = h / cell;
    var m: u32 = 0;
    for cy in 0..ch {
        for cx in 0..cw {
            if skip[cy * cw + cx] != 0 { continue; }
            let x0 = max(cx * cell, border);
            let x1 = min((cx + 1) * cell, w - border);
            let y0 = max(cy * cell, border);
            let y1 = min((cy + 1) * cell, h - border);
            var best = 0.0;
            var bx: u32 = 0;
            var by: u32 = 0;
            var y = y0;
            while y < y1 {
                var x = x0;
                while x < x1 {
                    let i = y * w + x;
                    let a = sxx[i];
                    let b = sxy[i];
                    let c = syy[i];
                    let s = 0.5 * (a + c) - sqrt(0.25 * (a - c) * (a - c) + b * b);
                    if s > best {
                        best = s;
                        bx = x;
                        by = y;
                    }
                    x += 1;
                }
                y += 1;
            }
            if best > 0.0 {
                out[3 * m] = bx as f32;
                out[3 * m + 1] = by as f32;
                out[3 * m + 2] = best / 25.0;
                m += 1;
            }
        }
    }
    return m;
}

// ------------------------------------------------------------------ light

// The light in a camera frame, so the scene can be drawn to match it: the mean red, green and blue
// of the whole frame, and of the box (x0, y0)-(x1, y1) (the floor under the scene), over every
// second pixel in each direction. out: r, g, b of the frame, r, g, b of the box (0-255), pixels in
// the box.
export fn light_stats(rgba: *u8, w: u32, h: u32, x0: u32, y0: u32, x1: u32, y1: u32, out: *f32) {
    var s = vec3(0.0);
    var sb = vec3(0.0);
    var n: u32 = 0;
    var nb: u32 = 0;
    var y: u32 = 0;
    while y < h {
        var x: u32 = 0;
        while x < w {
            let p = rgba + (y * w + x) * 4;
            let c = vec3(p[0] as f32, p[1] as f32, p[2] as f32);
            s += c;
            n += 1;
            if x >= x0 && x < x1 && y >= y0 && y < y1 {
                sb += c;
                nb += 1;
            }
            x += 2;
        }
        y += 2;
    }
    let m = s / max(n as f32, 1.0);
    let mb = sb / max(nb as f32, 1.0);
    out[0] = m.x;
    out[1] = m.y;
    out[2] = m.z;
    out[3] = mb.x;
    out[4] = mb.y;
    out[5] = mb.z;
    out[6] = nb as f32;
}

// ------------------------------------------------------------------ whole-image motion

// The image minus its local mean (7 x 7 box): texture without the lighting, so frames taken with
// different exposures can be compared directly. Borders (3 pixels) are 0. tmp has w * h entries.
export fn highpass(img: *u8, w: u32, h: u32, out: *f32, tmp: *f32) {
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = 0.0;
        }
    }
    for y in 0..h {
        let r = img + y * w;
        for x in 3..w - 3 {
            var s: u32 = 0;
            for k: u32 in 0..7 { s += r[x - 3 + k] as u32; }
            tmp[y * w + x] = s as f32;
        }
    }
    for y in 3..h - 3 {
        for x in 3..w - 3 {
            var s = 0.0;
            for k: u32 in 0..7 { s += tmp[(y - 3 + k) * w + x]; }
            out[y * w + x] = img[y * w + x] as f32 - s / 49.0;
        }
    }
}

// How far the whole image moved from a to b (high-passed images of one pyramid level): the shift
// d, within `range` pixels of (gx, gy), for which b(x + d) matches a(x) best (mean absolute
// difference over the middle of the image), refined to a fraction of a pixel. out: dx, dy, the best
// cost, and the mean cost of all shifts (a clear match costs much less than the mean). Returns
// false if the images overlap too little.
export fn global_shift(a: *f32, b: *f32, w: u32, h: u32, gx: i32, gy: i32, range: i32, out: *f32) -> bool {
    let wi = w as i32;
    let hi = h as i32;
    let x0 = max(4, 4 - (gx - range));
    let x1 = min(wi - 4, wi - 4 - (gx + range));
    let y0 = max(4, 4 - (gy - range));
    let y1 = min(hi - 4, hi - 4 - (gy + range));
    if x1 - x0 < wi / 3 || y1 - y0 < hi / 3 { return false; }
    let side = 2 * range + 1;
    var best = 3.0e38;
    var bi: i32 = 0;
    var bj: i32 = 0;
    var total = 0.0;
    var costs: [441]f32;              // up to range 10
    for j in 0..side {
        for i in 0..side {
            let dx = gx - range + i;
            let dy = gy - range + j;
            var s = 0.0;
            var y = y0;
            while y < y1 {
                let ra = a + (y * wi) as u32;
                let rb = b + ((y + dy) * wi + dx) as u32;
                var x = x0;
                while x < x1 {
                    s += abs(ra[x as u32] - rb[x as u32]);
                    x += 2;
                }
                y += 2;
            }
            costs[j * side + i] = s;
            total += s;
            if s < best {
                best = s;
                bi = i;
                bj = j;
            }
        }
    }
    // parabola through the best cost and its neighbours, along x and along y
    var fx = 0.0;
    var fy = 0.0;
    if bi > 0 && bi < side - 1 {
        let l = costs[bj * side + bi - 1];
        let r = costs[bj * side + bi + 1];
        let den = l - 2.0 * best + r;
        if den > 0.0 { fx = clamp(0.5 * (l - r) / den, -0.5, 0.5); }
    }
    if bj > 0 && bj < side - 1 {
        let u = costs[(bj - 1) * side + bi];
        let d = costs[(bj + 1) * side + bi];
        let den = u - 2.0 * best + d;
        if den > 0.0 { fy = clamp(0.5 * (u - d) / den, -0.5, 0.5); }
    }
    out[0] = (gx - range + bi) as f32 + fx;
    out[1] = (gy - range + bj) as f32 + fy;
    out[2] = best;
    out[3] = total / ((side * side) as f32);
    return true;
}

// ------------------------------------------------------------------ optical flow

// Pyramidal Lucas-Kanade (Bouguet's formulation) with an 11 x 11 window and a brightness offset
// per window, so a change of exposure between frames doesn't move the points.
// src: n points in `prev` (level-0 pixels); dst: a guess of where each is in `cur` on the way in,
// where it is on the way out. status: 1 if followed, 0 if lost. err: the RMS grey-level difference
// left between the two windows.
export fn track_points(prev: *u8, cur: *u8, w: u32, h: u32, levels: u32, n: u32,
                       src: *f32, dst: *f32, status: *u8, err: *f32) {
    var patch: [169]f32;              // 13 x 13: the window plus a border for the gradients
    var jw: [121]f32;                 // 11 x 11 window in the current frame
    var tv: [121]f32;
    var gx: [121]f32;
    var gy: [121]f32;
    let nn = 121.0;
    for i in 0..n {
        let p0 = vec2(src[2 * i], src[2 * i + 1]);
        var d = vec2(dst[2 * i], dst[2 * i + 1]) - p0;       // displacement in level-0 pixels
        var ok = true;
        var rms = 0.0;
        var l = levels;
        while l > 0 && ok {
            l -= 1;
            let sc = 1.0 / ((1 as u32 << l) as f32);
            let lw = w >> l;
            let lh = h >> l;
            let pim = prev + level_offset(w, h, l);
            let cim = cur + level_offset(w, h, l);
            let pl = (p0 + 0.5) * sc - 0.5;                // the point at this level
            let lo = pl - 6.0;
            let hi = pl + 6.0;
            if lo.x < 0.0 || lo.y < 0.0 || hi.x > (lw - 2) as f32 || hi.y > (lh - 2) as f32 {
                if l == 0 { ok = false; }
                continue;                                  // near an edge: try the next level
            }
            window(pim, lw, lo.x, lo.y, 13, &patch[0]);
            var sxx = 0.0;
            var sxy = 0.0;
            var syy = 0.0;
            var sgx = 0.0;
            var sgy = 0.0;
            for jy in 0..11 {
                for jx in 0..11 {
                    let k = jy * 11 + jx;
                    let c = (jy + 1) * 13 + jx + 1;
                    let ggx = 0.5 * (patch[c + 1] - patch[c - 1]);
                    let ggy = 0.5 * (patch[c + 13] - patch[c - 13]);
                    gx[k] = ggx;
                    gy[k] = ggy;
                    tv[k] = patch[c];
                    sxx += ggx * ggx;
                    sxy += ggx * ggy;
                    syy += ggy * ggy;
                    sgx += ggx;
                    sgy += ggy;
                }
            }
            // gradient matrix with the brightness offset solved out
            let a = sxx - sgx * sgx / nn;
            let b = sxy - sgx * sgy / nn;
            let c2 = syy - sgy * sgy / nn;
            let mineig = 0.5 * (a + c2) - sqrt(0.25 * (a - c2) * (a - c2) + b * b);
            if mineig < 0.3 * nn {
                if l == 0 { ok = false; }
                continue;                                  // too flat to follow at this level
            }
            let det = a * c2 - b * b;
            for it in 0..12 {
                let q = pl + d * sc;
                let qlo = q - 5.0;
                let qhi = q + 5.0;
                if qlo.x < 0.0 || qlo.y < 0.0 || qhi.x > (lw - 2) as f32 || qhi.y > (lh - 2) as f32 {
                    ok = false;
                    break;
                }
                window(cim, lw, qlo.x, qlo.y, 11, &jw[0]);
                var se = 0.0;
                var se2 = 0.0;
                var sex = 0.0;
                var sey = 0.0;
                for k in 0..121 {
                    let e = tv[k] - jw[k];
                    se += e;
                    se2 += e * e;
                    sex += e * gx[k];
                    sey += e * gy[k];
                }
                let mean = se / nn;
                let bx = sex - mean * sgx;
                let by = sey - mean * sgy;
                let dx = (c2 * bx - b * by) / det;
                let dy = (a * by - b * bx) / det;
                d += vec2(dx, dy) / sc;
                rms = sqrt(max(0.0, se2 / nn - mean * mean));
                if dx * dx + dy * dy < 0.0004 { break; }
            }
        }
        let p1 = p0 + d;
        dst[2 * i] = p1.x;
        dst[2 * i + 1] = p1.y;
        status[i] = select(ok, 1 as u8, 0 as u8);
        err[i] = rms;
    }
}

// ------------------------------------------------------------------ camera pose

// A camera pose: world -> camera rotation (its rows) and translation, so a world point X is at
// R X + t in camera coordinates (x right, y down, z forward).
struct Pose { r0: vec3, r1: vec3, r2: vec3, t: vec3 }

// prior: rotation (rows) the solution is pulled toward, with weight `prior_weight` (about how many
// points' worth); huber: residual (normalized image units) above which a point counts less.
struct PoseParams { p0: vec3, p1: vec3, p2: vec3, prior_weight: f32, huber: f32, iters: u32 }

fn rodrigues(v: vec3) -> mat3 {
    let th = length(v);
    if th < 1.0e-6 {
        return mat3(vec3(1.0, v.z, -v.y), vec3(-v.z, 1.0, v.x), vec3(v.y, -v.x, 1.0));
    }
    let k = v / th;
    let c = cos(th);
    let s = sin(th);
    let kx = mat3(vec3(0.0, k.z, -k.y), vec3(-k.z, 0.0, k.x), vec3(k.y, -k.x, 0.0));
    return mat3(c) + kx * s + mat3(k * k.x, k * k.y, k * k.z) * (1.0 - c);
}

// Gauss-Newton on the reprojection error of n points with known world positions xyz, seen at obs
// (normalized image coordinates: ((u - cx) / f, (v - cy) / f)), weights wts (0 = ignore), starting
// from *pose and writing the result there. resid: each point's final reprojection error
// (normalized units; 1e9 behind the camera). Returns the iterations run.
export fn solve_pose(n: u32, obs: *f32, xyz: *f32, wts: *f32, pose: *Pose, prm: *PoseParams,
                     resid: *f32) -> u32 {
    var rot = transpose(mat3(pose.r0, pose.r1, pose.r2));
    var t = pose.t;
    let prior = transpose(mat3(prm.p0, prm.p1, prm.p2));
    let hub = prm.huber;
    var iters: u32 = 0;
    for it in 0..prm.iters {
        var hm: [36]f32;
        var g: [6]f32;
        var ju: [6]f32;
        var jv: [6]f32;
        for i in 0..n {
            let wi = wts[i];
            if wi <= 0.0 { continue; }
            let x = vec3(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
            let c = rot * x + t;
            if c.z < 0.05 { continue; }
            let iz = 1.0 / c.z;
            let ru = c.x * iz - obs[2 * i];
            let rv = c.y * iz - obs[2 * i + 1];
            let e = sqrt(ru * ru + rv * rv);
            var wgt = wi;
            if e > hub { wgt = wi * hub / e; }
            let a = vec3(iz, 0.0, -c.x * iz * iz);        // d(x/z)/dc
            let b = vec3(0.0, iz, -c.y * iz * iz);        // d(y/z)/dc
            let ra = cross(c, a);                          // d(x/z)/d(rotation)
            let rb = cross(c, b);
            ju[0] = ra.x; ju[1] = ra.y; ju[2] = ra.z; ju[3] = a.x; ju[4] = a.y; ju[5] = a.z;
            jv[0] = rb.x; jv[1] = rb.y; jv[2] = rb.z; jv[3] = b.x; jv[4] = b.y; jv[5] = b.z;
            for r in 0..6 {
                g[r] += wgt * (ju[r] * ru + jv[r] * rv);
                for s in 0..r + 1 {
                    hm[r * 6 + s] += wgt * (ju[r] * ju[s] + jv[r] * jv[s]);
                }
            }
        }
        // pull toward the prior rotation: residual = the small rotation from here to the prior
        let m = prior * transpose(rot);
        let phi = 0.5 * vec3(m[1].z - m[2].y, m[2].x - m[0].z, m[0].y - m[1].x);
        let pw = prm.prior_weight;
        hm[0] += pw;
        hm[7] += pw;
        hm[14] += pw;
        g[0] -= pw * phi.x;
        g[1] -= pw * phi.y;
        g[2] -= pw * phi.z;
        // solve hm d = -g (Cholesky of the lower triangle, a little damping)
        var lm: [36]f32;
        var good = true;
        for r in 0..6 {
            for s in 0..r + 1 {
                var sum = hm[r * 6 + s];
                if r == s { sum = sum * 1.0001 + 1.0e-9; }
                for k in 0..s { sum -= lm[r * 6 + k] * lm[s * 6 + k]; }
                if r == s {
                    if sum <= 0.0 { good = false; }
                    lm[r * 6 + r] = sqrt(max(sum, 1.0e-12));
                } else {
                    lm[r * 6 + s] = sum / lm[s * 6 + s];
                }
            }
        }
        if !good { break; }
        var y: [6]f32;
        for r in 0..6 {
            var sum = -g[r];
            for k in 0..r { sum -= lm[r * 6 + k] * y[k]; }
            y[r] = sum / lm[r * 6 + r];
        }
        var d: [6]f32;
        for q in 0..6 {
            let r = 5 - q;
            var sum = y[r];
            for k in r + 1..6 { sum -= lm[k * 6 + r] * d[k]; }
            d[r] = sum / lm[r * 6 + r];
        }
        let dth = vec3(d[0], d[1], d[2]);
        let dt = vec3(d[3], d[4], d[5]);
        let e = rodrigues(dth);
        rot = e * rot;
        t = e * t + dt;
        iters += 1;
        if dot(dth, dth) + dot(dt, dt) < 1.0e-12 { break; }
    }
    let rows = transpose(rot);
    pose.r0 = rows[0];
    pose.r1 = rows[1];
    pose.r2 = rows[2];
    pose.t = t;
    for i in 0..n {
        let x = vec3(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
        let c = rot * x + t;
        if c.z < 0.05 {
            resid[i] = 1.0e9;
        } else {
            let ru = c.x / c.z - obs[2 * i];
            let rv = c.y / c.z - obs[2 * i + 1];
            resid[i] = sqrt(ru * ru + rv * rv);
        }
    }
    return iters;
}
