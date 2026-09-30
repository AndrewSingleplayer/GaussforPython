// Gaussian splatting renderer in HA++ (for AR on phones).
//
// Pipeline per frame (all on the GPU, see render_frame in the host examples):
//   1. preprocess  - project every splat, cull, compute 2D shape + color + tiles
//   2. scan_*      - prefix sum of "tiles touched" -> where each splat's work goes
//   3. bin         - one (tile, depth) key per splat per tile it covers
//   4. radix_*     - GPU radix sort of the keys (4 passes x 8 bits)
//   5. ranges      - where each 16x16 tile's splats start and end
//   6. render      - each tile blends its splats front to back, per pixel
//
// Loading (once): pack_splats() turns full-precision splats (as stored in .ply
// files, ~240 bytes each with full SH) into 32-byte PackedSplats: position, color+opacity
// as RGBA8, and the 3D covariance as six f16 values scaled by one f32. That is the main
// memory and bandwidth saving on phones.
//
// v2 (research/02-speed.md): each splat covers only the tiles of the bounding box of the
// ellipse where it adds at least 1/255 to a pixel (the same test render() applies per pixel),
// instead of a 3-sigma square: 1.2-3x fewer (splat, tile) pairs to sort on real captures,
// and no pixel changes. The covariance is stored divided by its largest variance, which keeps
// f16 precise (raw f16 loses 78-93% of the splats of real captures, research/01-splat-math.md).
//
// Camera convention: camera space is x right, y down, z forward (like the
// original 3DGS / OpenCV). ARKit/ARCore views are x right, y up, z backward:
// flip the y and z rows of their view matrix before filling Camera.view.

const TILE: u32 = 16;
const SH_C0: f32 = 0.28209479177387814;

// ---------------------------------------------------------------- data

// One splat as trained (the .ply layout, spherical harmonics degree 0 only).
struct GaussianRaw {
    pos: vec3,
    log_scale: vec3,
    rot: vec4,          // quaternion (w, x, y, z), not normalized
    opacity_logit: f32,
    sh_dc: vec3,        // f_dc_0..2
}

// 32 bytes per splat, what the GPU reads every frame.
struct PackedSplat {
    pos: vec3,
    color: u32,         // RGBA8: base color + opacity
    cov: [3]u32,        // 3D covariance / cov_scale: xx,xy | xz,yy | yz,zz as f16 pairs
    cov_scale: f32,     // the largest variance of the splat
}

struct Camera {
    view: mat4,         // world -> camera
    focal: vec2,        // fx, fy in pixels
    center: vec2,       // cx, cy in pixels
    size: vec2,         // width, height in pixels
    near: f32,
    far: f32,
    tiles_x: u32,
    tiles_y: u32,
}

// Result of preprocessing one splat.
struct Proj {
    center: vec2,       // pixel position
    conic: vec3,        // inverse 2D covariance (a, b, c)
    color: u32,         // RGBA8
    rect: uvec4,        // tiles covered: x0, y0, x1, y1 (end exclusive)
    depth: u32,         // 16-bit depth key
}

// ---------------------------------------------------------------- loading (CPU)

fn covariance(log_scale: vec3, rot: vec4) -> mat3 {
    let r = quat_to_mat3(rot);
    let s = exp(log_scale);
    let m = r * mat3(vec3(s.x, 0.0, 0.0), vec3(0.0, s.y, 0.0), vec3(0.0, 0.0, s.z));
    return m * transpose(m);
}

export fn pack_splats(raw: *GaussianRaw, out: *PackedSplat, n: i64) {
    for i in 0..n {
        let g = raw[i];
        let full = covariance(g.log_scale, g.rot);
        let scale = max(max(full[0].x, full[1].y), max(full[2].z, 1.0e-30));
        let cov = full * (1.0 / scale);
        let rgb = clamp(g.sh_dc * SH_C0 + 0.5, 0.0, 1.0);
        var p: PackedSplat;
        p.pos = g.pos;
        p.color = pack_unorm4(vec4(rgb, sigmoid(g.opacity_logit)));
        p.cov[0] = pack_half2(vec2(cov[0].x, cov[0].y));
        p.cov[1] = pack_half2(vec2(cov[0].z, cov[1].y));
        p.cov[2] = pack_half2(vec2(cov[1].z, cov[2].z));
        p.cov_scale = scale;
        out[i] = p;
    }
}

// ---------------------------------------------------------------- 1. preprocess

@workgroup(128)
kernel preprocess(splats: *PackedSplat, cams: *Camera, projs: *Proj, counts: *u32, offsets: *u32, n: u32) {
    let i = global_id.x;
    if i >= n { return; }
    counts[i] = 0;
    offsets[i] = 0;
    let s = splats[i];
    let cam = cams[0];
    let pc = cam.view * vec4(s.pos, 1.0);
    if pc.z < cam.near || pc.z > cam.far { return; }

    // 3D covariance -> camera space -> 2D (EWA splatting, as in 3DGS)
    let c0 = unpack_half2(s.cov[0]);
    let c1 = unpack_half2(s.cov[1]);
    let c2 = unpack_half2(s.cov[2]);
    let sigma = mat3(vec3(c0.x, c0.y, c1.x), vec3(c0.y, c1.y, c2.x), vec3(c1.x, c2.x, c2.y)) * s.cov_scale;
    let w = mat3(cam.view[0].xyz, cam.view[1].xyz, cam.view[2].xyz);
    let m = w * sigma * transpose(w);
    let lim = 1.3 * cam.size * 0.5 / cam.focal;
    let tx = clamp(pc.x / pc.z, -lim.x, lim.x) * pc.z;
    let ty = clamp(pc.y / pc.z, -lim.y, lim.y) * pc.z;
    let z2 = pc.z * pc.z;
    let j0 = vec3(cam.focal.x / pc.z, 0.0, -cam.focal.x * tx / z2);
    let j1 = vec3(0.0, cam.focal.y / pc.z, -cam.focal.y * ty / z2);
    let mj0 = m * j0;
    let mj1 = m * j1;
    let a = dot(j0, mj0) + 0.3;
    let b = dot(j0, mj1);
    let c = dot(j1, mj1) + 0.3;
    let det = a * c - b * b;
    if det <= 0.0 { return; }
    // render() skips a pixel where opacity * exp(-q/2) < 1/255 (q = d^T conic d), so the splat only
    // reaches pixels inside the ellipse q <= cut. Its bounding box has half-widths sqrt(cut * a),
    // sqrt(cut * c) (a, c: the diagonal of the 2D covariance).
    let opacity = ((s.color >> 24) as f32) * 0.00392156862745098;
    let cut = 2.0 * log(max(255.0 * opacity, 1.0));
    if cut <= 0.0 { return; }
    let half = sqrt(cut * vec2(a, c));

    let center = vec2(cam.focal.x * pc.x / pc.z + cam.center.x, cam.focal.y * pc.y / pc.z + cam.center.y);
    let tiles = vec2(cam.tiles_x as f32, cam.tiles_y as f32);
    let lo = clamp(floor((center - half) / 16.0), vec2(0.0), tiles);
    let hi = clamp(floor((center + half) / 16.0) + 1.0, vec2(0.0), tiles);
    let rect = uvec4(lo.x as u32, lo.y as u32, hi.x as u32, hi.y as u32);
    let count = (rect.z - rect.x) * (rect.w - rect.y);
    if count == 0 { return; }

    var p: Proj;
    p.center = center;
    p.conic = vec3(c, -b, a) / det;
    p.color = s.color;
    p.rect = rect;
    let dz = clamp((pc.z - cam.near) / (cam.far - cam.near), 0.0, 1.0);
    p.depth = (dz * 65535.0) as u32;
    projs[i] = p;
    counts[i] = count;
    offsets[i] = count;     // becomes the exclusive prefix sum after scan_*
}

// ---------------------------------------------------------------- 2. prefix sums

// Exclusive prefix sum of blocks of 256 values, in place; block totals -> sums.
@workgroup(256)
kernel scan_blocks(data: *u32, sums: *u32, n: u32) {
    shared tmp: [256]u32;
    let i = global_id.x;
    let t = local_id.x;
    var v: u32 = 0;
    if i < n { v = data[i]; }
    tmp[t] = v;
    barrier();
    var off: u32 = 1;
    while off < 256 {
        var add: u32 = 0;
        if t >= off { add = tmp[t - off]; }
        barrier();
        tmp[t] += add;
        barrier();
        off = off * 2;
    }
    if i < n { data[i] = tmp[t] - v; }
    if t == 255 { sums[group_id.x] = tmp[255]; }
}

// Exclusive prefix sum of the block totals (any count), by one workgroup.
@workgroup(256)
kernel scan_sums(sums: *u32, nblocks: u32) {
    shared tmp: [256]u32;
    shared carry: [1]u32;
    let t = local_id.x;
    if t == 0 { carry[0] = 0; }
    barrier();
    var base: u32 = 0;
    while base < nblocks {
        let i = base + t;
        var v: u32 = 0;
        if i < nblocks { v = sums[i]; }
        tmp[t] = v;
        barrier();
        var off: u32 = 1;
        while off < 256 {
            var add: u32 = 0;
            if t >= off { add = tmp[t - off]; }
            barrier();
            tmp[t] += add;
            barrier();
            off = off * 2;
        }
        let c = carry[0];
        if i < nblocks { sums[i] = c + tmp[t] - v; }
        barrier();
        if t == 255 { carry[0] = c + tmp[255]; }
        barrier();
        base += 256;
    }
}

@workgroup(256)
kernel scan_add(data: *u32, sums: *u32, n: u32) {
    let i = global_id.x;
    if i < n { data[i] += sums[group_id.x]; }
}

// ---------------------------------------------------------------- 3. binning

@workgroup(128)
kernel bin(projs: *Proj, offsets: *u32, counts: *u32, keys: *u32, vals: *u32, n: u32, tiles_x: u32) {
    let i = global_id.x;
    if i >= n { return; }
    if counts[i] == 0 { return; }
    let p = projs[i];
    var o = offsets[i];
    for y in p.rect.y..p.rect.w {
        for x in p.rect.x..p.rect.z {
            keys[o] = ((y * tiles_x + x) << 16) | p.depth;
            vals[o] = i;
            o += 1;
        }
    }
}

// ---------------------------------------------------------------- 4. radix sort

// Per 256-key block: how many keys have each 8-bit digit. Layout hist[digit * nblocks + block].
@workgroup(256)
kernel radix_hist(keys: *u32, hist: *u32, n: u32, shift: u32, nblocks: u32) {
    shared counts: [256]u32;
    let t = local_id.x;
    counts[t] = 0;
    barrier();
    let i = global_id.x;
    if i < n { atomic_add(counts, (keys[i] >> shift) & 255, 1); }
    barrier();
    hist[t * nblocks + group_id.x] = counts[t];
}

// Stable scatter: out position = scanned hist offset + rank among equal digits earlier in the block.
@workgroup(256)
kernel radix_scatter(keys: *u32, vals: *u32, keys_out: *u32, vals_out: *u32, hist: *u32,
                     n: u32, shift: u32, nblocks: u32) {
    shared digits: [256]u32;
    let t = local_id.x;
    let i = global_id.x;
    var d: u32 = 256;
    if i < n { d = (keys[i] >> shift) & 255; }
    digits[t] = d;
    barrier();
    if i < n {
        var rank: u32 = 0;
        for j in 0..t { if digits[j] == d { rank += 1; } }
        let dst = hist[d * nblocks + group_id.x] + rank;
        keys_out[dst] = keys[i];
        vals_out[dst] = vals[i];
    }
}

// ---------------------------------------------------------------- 5. tile ranges

@workgroup(256)
kernel clear(data: *u32, n: u32) {
    let i = global_id.x;
    if i < n { data[i] = 0; }
}

@workgroup(256)
kernel ranges(keys: *u32, spans: *uvec2, n: u32) {
    let i = global_id.x;
    if i >= n { return; }
    let tile = keys[i] >> 16;
    if i == 0 || (keys[i - 1] >> 16) != tile { spans[tile].x = i; }
    if i == n - 1 || (keys[i + 1] >> 16) != tile { spans[tile].y = i + 1; }
}

// ---------------------------------------------------------------- 6. render

@workgroup(16, 16)
kernel render(projs: *Proj, vals: *u32, spans: *uvec2, image: *u32, width: u32, height: u32, background: u32) {
    shared sx: [256]f32;
    shared sy: [256]f32;
    shared ca: [256]f32;
    shared cb: [256]f32;
    shared cc: [256]f32;
    shared col: [256]u32;
    let px = global_id.x;
    let py = global_id.y;
    let inside = px < width && py < height;
    let t = local_id.x + local_id.y * 16;
    let span = spans[group_id.x + group_id.y * num_groups.x];
    let pixel = vec2(px as f32, py as f32);
    var trans = 1.0;
    var rgb = vec3(0.0);
    var done = !inside;
    var start = span.x;
    while start < span.y {
        let k = start + t;
        if k < span.y {
            let p = projs[vals[k]];
            sx[t] = p.center.x;
            sy[t] = p.center.y;
            ca[t] = p.conic.x;
            cb[t] = p.conic.y;
            cc[t] = p.conic.z;
            col[t] = p.color;
        }
        barrier();
        let count = min(256, span.y - start);
        for j in 0..count {
            if !done {
                let d = vec2(sx[j], sy[j]) - pixel;
                let power = -0.5 * (ca[j] * d.x * d.x + cc[j] * d.y * d.y) - cb[j] * d.x * d.y;
                if power <= 0.0 {
                    let c = unpack_unorm4(col[j]);
                    let alpha = min(0.99, c.w * exp(power));
                    if alpha >= 0.00392156862745098 {
                        let next = trans * (1.0 - alpha);
                        if next < 0.0001 {
                            done = true;
                        } else {
                            rgb += c.xyz * (alpha * trans);
                            trans = next;
                        }
                    }
                }
            }
        }
        barrier();
        start += 256;
    }
    if inside {
        let bg = unpack_unorm4(background).xyz;
        image[py * width + px] = pack_unorm4(vec4(rgb + bg * trans, 1.0));
    }
}
