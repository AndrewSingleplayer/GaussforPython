# 1. The math of Gaussian splatting

What every renderer in this repository computes (`gaussian/splat.ha`, `web/src/viewer.js`),
written out once with the reasons behind each step. Notation: vectors are columns, `Σ` is a
covariance matrix, `W` is the world→camera rotation, `f_x, f_y` are focal lengths in pixels.

## A scene is a set of 3D Gaussians

Each splat has:

| Stored as (3DGS `.ply`) | Used as |
|---|---|
| position `μ` (3 floats) | centre of the Gaussian |
| `log_scale` (3) | standard deviations `s = exp(log_scale)` along the splat's own axes |
| rotation quaternion `q = (w, x, y, z)` (4, not normalized) | `R = quat_to_mat3(q / |q|)` |
| `opacity_logit` (1) | opacity `o = sigmoid(opacity_logit)` |
| spherical harmonics, `(d+1)²` × RGB (3 to 48 floats) | colour as a function of view direction |

The Gaussian's density is `G(x) = exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))` with

    Σ = R S Sᵀ Rᵀ,   S = diag(s)

Storing `R` and `S` rather than `Σ` keeps `Σ` positive semi-definite during training. A
renderer can precompute `Σ` once per splat (six unique numbers).

## Projection: EWA splatting

A perspective projection turns a 3D Gaussian into something that isn't exactly Gaussian. EWA
splatting (Zwicker et al., 2001) linearizes the projection at the splat's centre. `t = W μ + t_w`
is the centre in camera space (x right, y down, z forward). The Jacobian of `(x, y, z) ↦
(f_x x/z, f_y y/z)` there is

    J = | f_x/z    0      -f_x x/z² |
        |   0    f_y/z    -f_y y/z² |

and the screen-space covariance is

    Σ' = J W Σ Wᵀ Jᵀ      (2×2)

3DGS clamps `x/z` and `y/z` to 1.3× the field of view inside `J` (not for the centre). This
keeps splats near the screen edge from being stretched without limit by the linearization.

## Anti-aliasing term

3DGS adds `0.3` to the diagonal of `Σ'`: a Gaussian of variance 0.3 px² convolved with every
splat, so no splat is thinner than about a pixel. Mip-Splatting (Yu et al., CVPR 2024) shows
this "dilation" is what makes captures look too thick when zoomed out and too thin when zoomed
in. It replaces the term with a 2D Mip filter (variance ≈ 0.1, approximating the pixel's box
filter) and scales the opacity by `sqrt(det Σ' / det(Σ' + 0.1 I))`, so small splats keep their
total energy. **A model must be rendered with the filter it was trained with.** The six
captures here are standard 3DGS, so every renderer here uses `+0.3` and no compensation.

## Per pixel: the conic and alpha

For a pixel at offset `d` from the projected centre:

    power = -½ dᵀ Σ'⁻¹ d,     α = min(0.99, o · exp(power))

`Σ'⁻¹` is the "conic" `(a, b, c)` = `(Σ'₂₂, -Σ'₁₂, Σ'₁₁) / det Σ'`. The 0.99 cap keeps a single
splat from making a pixel fully opaque. That matters for early termination and for training
gradients.

## Compositing

Splats sorted front to back give each pixel

    C = Σᵢ cᵢ αᵢ Tᵢ,   Tᵢ = Πⱼ<ᵢ (1 - αⱼ)

3DGS stops a pixel when `T < 10⁻⁴`. Drawing back to front with premultiplied alpha
(`dst = src + (1 - α_src) dst`) gives the same `C` exactly. The web viewer does this with the
GPU's blending hardware.

## How far to draw a splat

The v1 compute renderer bounds each splat by a square of radius `3 √λ_max`, where `λ` are the
eigenvalues of `Σ'`:

    λ = mid ± sqrt(mid² - det),   mid = ½ (Σ'₁₁ + Σ'₂₂)

A splat changes a pixel only while `o · exp(power) ≥ 1/255`, that is while

    dᵀ Σ'⁻¹ d ≤ 2 ln(255 o)

That region is an ellipse with semi-axes `k √λ₁` and `k √λ₂`, where `k = sqrt(2 ln(255 o))`,
along the eigenvectors of `Σ'`. For `o = 0.5`, `k = 3.0`; for `o = 0.05`, `k = 1.6`. The web
viewer draws exactly this oriented rectangle. [`02-speed.md`](02-speed.md) measures what that
saves on the real scenes.

## Colour: spherical harmonics

Colour depends on the direction `v` from the camera to the splat:

    c(v) = 0.5 + Σₗ Σₘ kₗₘ Yₗₘ(v)       (clamped to 0..1; degree 0 alone: 0.5 + 0.2821 k₀₀)

Degree 0 is one constant (`SH_C0 = 0.28209479`). Degrees 1–3 add 3, 5 and 7 terms per channel
(45 floats in total). They carry highlights and reflections. How much they matter depends on the
scene: [`analyze_scenes.py`](analyze_scenes.py) measures it.

## Formats and their precision

| Format | Bytes per splat | Notes |
|---|---|---|
| 3DGS `.ply` | 248 (SH degree 3) | float32 everything |
| Niantic `.spz` | ~64 (SH3), 19 without SH, before gzip | 24-bit fixed-point positions, 8-bit log scale, 8-bit quaternion xyz |
| antimatter15 `.splat` | 32 | float32 position and scale, RGBA8, quaternion as 4 × u8 |
| HA++ v1 `PackedSplat` | 32 | position f32, RGBA8, covariance as 6 × f16 |
| HA++ web `.hspl` (download) | 16 | u16 position in the bounding box, RGBA8, 8-bit log scale, 8-bit quaternion xyz |
| HA++ web GPU texture | 32 | position f32, RGBA8, largest variance f32, covariance / that variance as 6 × f16 |

**f16 and the covariance.** The smallest normal f16 is 6.1·10⁻⁵. In real captures the
variance of a splat is often far below that: in 78–93% of splats at least one diagonal entry of
`Σ` is (measured on the six scenes). Stored raw as f16, those entries lose most of their bits
(subnormals) or become 0. The loss is invisible at a normal distance but shows in close-ups,
which is exactly how AR is viewed. Dividing `Σ` by its largest entry before the f16 conversion,
and storing that one number as f32, keeps every entry at full f16 relative precision. The web
viewer does this. The v1 `PackedSplat` doesn't yet (see [`README.md`](README.md)).

**Quantized quaternions.** With `x, y, z` stored in 8 bits and `w = sqrt(1 - x² - y² - z²)`,
rounding can push `x² + y² + z²` above 1. `w` is then 0, and the quaternion must be
renormalized, or `R` isn't a rotation and `Σ` comes out up to 4% too large. HA++'s
`quat_to_mat3` normalizes. `tests/test_web.py` found that the first version of the JavaScript
fallback didn't.

## Sources

- B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, *3D Gaussian Splatting for Real-Time
  Radiance Field Rendering*, SIGGRAPH 2023. https://arxiv.org/abs/2308.04079
- M. Zwicker, H. Pfister, J. van Baar, M. Gross, *EWA Splatting*, IEEE TVCG 2002 (Visualization 2001).
- Z. Yu, A. Chen, B. Huang, T. Sattler, A. Geiger, *Mip-Splatting: Alias-free 3D Gaussian
  Splatting*, CVPR 2024. https://arxiv.org/abs/2311.16493
- Niantic SPZ format: https://github.com/nianticlabs/spz
- antimatter15/splat: https://github.com/antimatter15/splat
