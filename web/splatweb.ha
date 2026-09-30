// CPU side of the web splat viewer, compiled to WebAssembly (`happ build web/splatweb.ha -t web`).
// It runs in a Web Worker, in Safari on iPhone and every other browser:
//
//   decode_splats  the 16-byte download format -> the 32-byte GPU format + a culling radius
//   sort_splats    view-frustum culling + back-to-front order (for alpha blending)
//
// Download format, 16 bytes per splat (written by web/pack.py), most important splats first:
//    0  u32      colour RGBA8 (base colour and opacity)
//    4  u16 x3   position, 0..65535 across the scene's bounding box
//   10  u8 x3    scale, log-quantized: exp(lmin + q / 255 * lrange)
//   13  u8 x3    rotation quaternion x, y, z as q / 127.5 - 1 (w = sqrt(1 - x^2 - y^2 - z^2) >= 0)
//
// GPU format, 32 bytes per splat (two RGBA32UI texels):
//    0  f32 x3   position
//   12  u32      colour RGBA8
//   16  f32      s = largest variance of the splat (its 3D covariance is s * C)
//   20  f16 x6   C = covariance / s: xx, xy, xz, yy, yz, zz
// Storing C = covariance / s keeps f16 precise: a raw covariance of a small splat is below
// the smallest normal f16 (6.1e-5) for about 90% of the splats in real captures.

struct Box { lo: vec3, size: vec3, lmin: f32, lrange: f32 }

// Camera for sorting: rows of the world -> camera rotation (x right, y down, z forward), the
// translation, tan of half the field of view in x and y, and the near plane.
struct SortCamera { r0: vec3, r1: vec3, r2: vec3, t: vec3, tan_x: f32, tan_y: f32, near: f32 }

export fn decode_splats(src: *u8, n: u32, box: *Box, gpu: *u32, pos: *f32, radius: *f32) {
    let b = box[0];
    for i in 0..n {
        let rec = src + i * 16;
        let q16 = (rec + 4) as *u16;
        let p = b.lo + vec3(q16[0] as f32, q16[1] as f32, q16[2] as f32) * (b.size / 65535.0);
        let s = exp(vec3(rec[10] as f32, rec[11] as f32, rec[12] as f32) * (b.lrange / 255.0) + b.lmin);
        let q = vec3(rec[13] as f32, rec[14] as f32, rec[15] as f32) / 127.5 - 1.0;
        let w = sqrt(max(0.0, 1.0 - dot(q, q)));
        let r = quat_to_mat3(vec4(w, q.x, q.y, q.z));
        let m = r * mat3(vec3(s.x, 0.0, 0.0), vec3(0.0, s.y, 0.0), vec3(0.0, 0.0, s.z));
        let cov = m * transpose(m);
        let big = max(s.x, max(s.y, s.z));
        let scale = big * big;
        let c = cov * (1.0 / scale);
        let o = i * 8;
        gpu[o] = f32_bits(p.x);
        gpu[o + 1] = f32_bits(p.y);
        gpu[o + 2] = f32_bits(p.z);
        gpu[o + 3] = (rec as *u32)[0];
        gpu[o + 4] = f32_bits(scale);
        gpu[o + 5] = pack_half2(vec2(c[0].x, c[0].y));
        gpu[o + 6] = pack_half2(vec2(c[0].z, c[1].y));
        gpu[o + 7] = pack_half2(vec2(c[1].z, c[2].z));
        pos[i * 3] = p.x;
        pos[i * 3 + 1] = p.y;
        pos[i * 3 + 2] = p.z;
        radius[i] = 3.0 * big;
    }
}

// Writes the indices of the splats (among the first n) that can be on screen to `order`, farthest
// first, and returns how many there are. keys and ids are scratch space (n entries each), counts
// has 65536 entries. Depths are quantized to 16 bits between the nearest and farthest visible splat.
export fn sort_splats(pos: *f32, radius: *f32, n: u32, cam: *SortCamera, keys: *u32, ids: *u32,
                      counts: *u32, order: *u32) -> u32 {
    let c = cam[0];
    let kx = sqrt(1.0 + c.tan_x * c.tan_x);     // distance to a side plane = (|x| - tan * z) / k
    let ky = sqrt(1.0 + c.tan_y * c.tan_y);
    let depth = keys as *f32;
    var m: u32 = 0;
    var zmin = 3.0e38;
    var zmax = -3.0e38;
    for i in 0..n {
        let p = vec3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
        let z = dot(c.r2, p) + c.t.z;
        let r = radius[i];
        let x = dot(c.r0, p) + c.t.x;
        let y = dot(c.r1, p) + c.t.y;
        // branch-free: splats are stored by importance, not by place, so whether the next one is
        // visible is close to random and a branch here would be mispredicted about half the time
        let visible = z > c.near && abs(x) - c.tan_x * z < r * kx && abs(y) - c.tan_y * z < r * ky;
        if visible {
            depth[m] = z;
            ids[m] = i;
            m += 1;
            zmin = min(zmin, z);
            zmax = max(zmax, z);
        }
    }
    if m == 0 { return 0; }
    for k in 0..65536 { counts[k] = 0; }
    let scale = 65535.0 / max(zmax - zmin, 1.0e-6);
    for j in 0..m {
        let key = ((zmax - depth[j]) * scale) as u32;     // 0 = farthest
        keys[j] = key;
        counts[key] += 1;
    }
    var sum: u32 = 0;
    for k in 0..65536 {
        let v = counts[k];
        counts[k] = sum;
        sum += v;
    }
    for j in 0..m {
        let key = keys[j];
        order[counts[key]] = ids[j];
        counts[key] += 1;
    }
    return m;
}
