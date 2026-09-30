# 3. Quality: what makes splats look wrong, and what to do about it

## Rendering a model the way it was trained

A 3DGS model is only correct under the renderer it was trained with. The six captures here are
standard 3DGS:
- `+0.3` px² dilation of the 2D covariance, without opacity compensation;
- the Jacobian clamped at 1.3× the field of view;
- alpha capped at 0.99;
- one depth per splat for sorting.

Every renderer in this repository does exactly that. Two "improvements" look tempting but would
make these models look worse, because the models were never trained for them:
- the Mip-Splatting 2D filter (a model trained with it needs it; a vanilla model gets thinner);
- the StopThePop per-pixel sort (it fixes popping; a vanilla model looks right with a global
  sort, which it was trained against).

## Aliasing when zooming (Mip-Splatting)

- **The problem.** Zoom out and splats that were trained to be a few pixels wide become thinner
  than a pixel. The 0.3 px² dilation then makes them brighter and thicker than they should be.
  Zoom in and the same dilation is too small, so fine splats look like sharp needles.
- **Mip-Splatting's fix** (Yu et al., CVPR 2024):
  - a 3D smoothing filter, limiting how small a splat may be to the sampling rate of the
    training views;
  - a 2D Mip filter: a Gaussian of variance about 0.1 px² that approximates the pixel's box
    filter, with the opacity scaled by `sqrt(det Σ' / det(Σ' + 0.1 I))`.
- **For a phone** this matters because AR means extreme zoom in (walking up to the object) and
  zoom out (the object across the room).
- **Plan.** Support the Mip filter as a per-scene flag for scenes trained with it. Keep vanilla
  behaviour for vanilla scenes.

## Popping (StopThePop)

- **The problem.** Sorting by one depth per splat means two overlapping splats can swap order
  when the camera turns slightly. Their blend then changes suddenly: popping.
- **StopThePop's fix** (Radl et al., SIGGRAPH 2024) is a hierarchical per-pixel sort. It is only
  4% slower than 3DGS. With models trained for it, about half as many splats give the same
  quality.
- **The web viewer** sorts by depth along the view axis, the same key 3DGS uses. The 16-bit
  quantization adds ties between splats less than `(zmax - zmin) / 65535` apart. A stable sort
  keeps their order consistent between frames. That order is the importance order, which
  doesn't depend on the view.

## View-dependent colour (spherical harmonics)

Measured with [`analyze_scenes.py`](analyze_scenes.py) on the two scenes that have SH degree 3:

| scene | energy of SH degrees 1–3 vs the variation of the base colours | split by degree 1 / 2 / 3 |
|---|---|---|
| raccoon family | 16.3% | 27% / 33% / 40% |
| unicorn plush | 0.2% | 20% / 33% / 47% |

- **Unicorn:** a matte plush has almost no view-dependent colour, so degree 0 is enough.
- **Raccoons:** glossy figurines have real highlights. Degree 1 alone (3 more coefficients per
  channel, 9 bytes at 8 bits) keeps about a quarter of that energy. Degree 3 needs 45 values.
- **Now:** both renderers and the web format store degree 0 only.
- **Next:** add degree 1 as an option in the web format (+9 bytes per splat), for scenes above
  a few percent.

## Precision

| What | Where | Effect | Status |
|---|---|---|---|
| 3D covariance as raw f16 | v1 `PackedSplat` | 78–93% of splats have a variance below the smallest normal f16. Fine detail blurs in close-ups. | web: fixed (scaled f16). v1: to do |
| 16-bit depth keys | web sort | ties within 1/65535 of the depth range | no visible effect seen |
| 8-bit log scale | web download format | 6% size steps | the same as Niantic's SPZ |
| 8-bit quaternion xyz | web download format | about 0.5° | renormalized when rounding pushes \|xyz\| past 1 |
| fragment math in f16 | web fragment shader | `exp(-½ r²)` for r up to 3.3 σ | well within f16 |

## Quantization vs the original

The web viewer reads 16-byte splats. The v1 path renders from the original float32 data. A
side-by-side check of the same camera with both paths is the next item. It needs a GPU image
comparison (the NumPy reference in `tests/splat_reference.py` can render both inputs).

## Sources

- Z. Yu et al., *Mip-Splatting: Alias-free 3D Gaussian Splatting*, CVPR 2024,
  https://arxiv.org/abs/2311.16493
- L. Radl et al., *StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time
  Rendering*, SIGGRAPH 2024 (ACM TOG 43(4)), https://arxiv.org/abs/2402.00525
- Niantic SPZ: https://github.com/nianticlabs/spz
