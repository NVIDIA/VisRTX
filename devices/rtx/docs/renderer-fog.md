# VISRTX_RENDERER_FOG: camera-surface depth cueing

`VISRTX_RENDERER_FOG` is a **VisRTX vendor extension**, advertised as
`ANARI_VISRTX_RENDERER_FOG`. It is RTX-only and introduces no new ANARI object.
Set its parameters on an existing Renderer.

## Support and discovery

Supported Renderer subtypes are `fast`, `interactive`, `default` (an alias for
`interactive`), and `quality`. Diagnostic `debug` (including its `debug_*`
factory aliases) and `test` Renderers do not support or advertise fog.

Query `anariGetObjectInfo(device, ANARI_DEVICE, nullptr, "extension",
ANARI_STRING_LIST)` for device availability, then query the same information
with `ANARI_RENDERER` and the intended subtype for subtype support. The device's
`extension` property also reports availability. A device-wide capability is
not a promise about every Renderer.

The VisRTX utility provides the same fog check:

```cpp
#include <anari/ext/visrtx/visrtx_extensions.h>

const auto features =
    visrtx::getObjectExtensions(device, ANARI_RENDERER, "quality");
if (features.VISRTX_RENDERER_FOG) {
  // Configure fog on a quality Renderer.
}
```

The generated `parameter` object-info list contains exactly the seven fog
parameters below on each supported subtype. `anariGetParameterInfo()` exposes
`default`, `description`, `required`, numeric `minimum`, selector `value`
(`ANARI_STRING_LIST`), and `sourceExtension` (`VISRTX_RENDERER_FOG`). Units,
finite-value constraints, and conditional applicability are in the descriptions;
there is no separate query for relational constraints such as end > start.
This contract uses subtype queries, not a new Renderer-instance property.

## Parameters

All parameters are optional. Defaults are restored when a parameter is unset
and the Renderer is committed.

| Parameter | ANARI type | Default | Meaning |
|---|---|---|---|
| `fogMode` | `ANARI_STRING` | `"none"` | `none`, `linear`, `exp`, `exp2`; `none` is the only disable switch |
| `fogDistanceMetric` | `ANARI_STRING` | `"viewDepth"` | `viewDepth` or `rayDistance`, in world units |
| `fogColorSource` | `ANARI_STRING` | `"constant"` | `constant` or `background` |
| `fogColor` | `ANARI_FLOAT32_VEC3` | `(1,1,1)` | Finite nonnegative **linear RGB**, including HDR values above one; used only with `constant` |
| `fogStart` | `ANARI_FLOAT32` | `0` | Finite nonnegative world distance; used only by `linear` |
| `fogEnd` | `ANARI_FLOAT32` | `1` | Finite world distance strictly greater than start; used only by `linear` |
| `fogDensity` | `ANARI_FLOAT32` | `1` | Finite nonnegative inverse-world-distance curve coefficient; used only by `exp` and `exp2` |

For surface RGB `S`, selected fog RGB `B`, distance `d`, visibility `T`, and fog
fraction `F = 1-T`, the camera appearance is `T*S + F*B`:

| Mode | Visibility / fraction |
|---|---|
| `none` | `T = 1`, `F = 0` |
| `linear` | `F = clamp((d-start)/(end-start), 0, 1)` |
| `exp` | `T = exp(-density*d)` |
| `exp2` | `T = exp(-(density*d)^2)` |

Zero density or zero distance yields no fog. When the exponential argument is
large enough that `exp(-x)` underflows, `T` is `0` and `F` is `1` (full fog).
A short valid linear interval is honored as given; no world-unit epsilon is
imposed. Density in `exp2` is an artistic curve coefficient, not a physical
homogeneous extinction coefficient.

### Camera metrics

Let `P` be the world-space surface point, `C` the committed camera position,
`V` its normalized forward direction, and `O` the **original camera-sample ray
origin**:

- `viewDepth`: `max(0, dot(P-C, V))`. Ray-buffer cameras still use their committed
  reference position/direction, not each supplied ray's direction.
- `rayDistance`: `length(P-O)`. `O` is the pinhole position, per-pixel
  orthographic origin, sampled lens origin for depth of field, or supplied
  ray-buffer origin. An orthographic plane does not acquire radial edge fog.

Clipping intervals, cutting planes, coverage pass-through, and numerical ray
re-origining do not change `O`. Distances are world units, not normalized device
depth, and do not depend on a far plane. Translating camera and scene together
preserves fog; scaling both by a positive factor preserves it when start/end
scale likewise or exponential density is divided by that factor.

### Precision and cost

Distance and curve evaluation run in double precision on the GPU, once per
camera-visible surface hit (never per bounce). This keeps accepted subnormal
`FLOAT32` densities and tiny visibilities exact under fast-math flush-to-zero,
and keeps small fog fractions meaningful against HDR fog colors. FP64 throughput
is low on consumer GPUs; the per-hit cost is a few dozen double operations and
has not been benchmarked separately. Diagnostic pipelines compile the fog path
out.

## Examples

Using the ANARI C++ helpers, with an already created supported `renderer`:

```cpp
// Constant-color linear depth cueing from 10 to 100 world units.
const float blueGray[] = {0.25f, 0.35f, 0.5f};
anari::setParameter(device, renderer, "fogMode", "linear");
anari::setParameter(device, renderer, "fogDistanceMetric", "viewDepth");
anari::setParameter(device, renderer, "fogColorSource", "constant");
anari::setParameter(device, renderer, "fogColor", ANARI_FLOAT32_VEC3, blueGray);
anari::setParameter(device, renderer, "fogStart", 10.f);
anari::setParameter(device, renderer, "fogEnd", 100.f);
anari::commitParameters(device, renderer);

// Fade toward the visible backdrop, using radial camera-sample distance.
anari::setParameter(device, renderer, "fogMode", "exp");
anari::setParameter(device, renderer, "fogDistanceMetric", "rayDistance");
anari::setParameter(device, renderer, "fogColorSource", "background");
anari::setParameter(device, renderer, "fogDensity", 0.01f); // inverse world units
anari::commitParameters(device, renderer);
// The earlier start/end and fogColor have no effect in this configuration.

anari::unsetParameter(device, renderer, "fogMode"); // restore "none"
anari::commitParameters(device, renderer);
```

### Visible background and linear color

`background` selects the existing **unfogged visible backdrop**. It respects
visible HDRI selection/addition, light tint and radiance scale, instance
transforms, and existing precedence over screen backgrounds. Invisible HDRIs
can illuminate surfaces without becoming fog color. Flat/image backgrounds use
the same frame-resolution screen coordinates as the ordinary compositor,
including image-region behavior. Directional environment lookup uses the
original camera-sample direction, never a reflection direction.

Fog reuses the effective linear RGB under `premultiplyBackground`; it does not
reinterpret background alpha or guess an external compositor's backdrop.
`fogColor` is entirely ignored in background mode, not used as a tint. The
operation happens before sample filtering, accumulation, output encoding, and
display transforms. For numerical comparisons disable denoising and nonlinear
firefly filtering; those existing image operations can change the final output.

## Coverage, transport, and channels

For coverage `alpha` and premultiplied RGB `Pm`, the deposited contribution is
`T*Pm + alpha*F*B`. **Alpha does not change.** Each directly visible layer uses
its own distance, before existing compositing/coverage decisions. Discarded or
zero-coverage contributions inject no fog color. Silhouette/depth-edge samples
are fogged individually, not by a postprocess over resolved color/depth.

Fast/interactive retain their straight-through coverage and colored material
transmission weights. Quality retains its stochastic coverage continuation:
finite-sample images need not equal deterministic layer compositing exactly.
The accepted camera surface's complete shaded result, including indirect
reflection/refraction, is depth-cued once. Secondary transport acquires no
additional fog events, and appearance weights do not change BSDFs, emission
used for lighting, light sampling/PDFs, shadows, random decisions, or Russian
roulette. Full fog does not terminate geometric traversal or skip metadata.

Already-visible analytic Light Proxies receive the opaque surface transform at
their hit distance. Fast still excludes these proxies. Visibility, unpickable
IDs, and subtype-specific depth behavior stay unchanged.

Scientific volumes are not fog. Direct volume RGB is unchanged, while directly
visible surfaces behind volume attenuation use their own surface distances and
existing visibility weights. After a volume scattering event, later surfaces
and lights are indirect transport, not new camera-fog events. Indirect volume
radiance belongs to the already depth-cued camera surface's shading.

Misses keep their existing backdrop color and alpha, even with full fog. Depth,
primitive/object/instance IDs, normals, and albedo retain their existing meanings
and sampling rules. The independent Renderer Pipelines, callable Material Shader
interfaces, Light Proxy representation, and separate surface/volume TLASes are
unchanged. Fog needs no new callable slot or acceleration-structure rebuild.

## Commit and invalid inputs

Parameter edits take effect at Commit through the normal Frontend lifecycle.
Unsetting any parameter restores its default, never a stale prior value.
Effective fog changes invalidate accumulated color; changes to an observed
background image/HDRI or background setting also invalidate dependent color.
Use the normal ANARI array map/unmap or object commit lifecycle for such edits.
Renderers may share a World without sharing fog state.

`none` bypasses validation of inactive selectors/numbers. Active modes validate
mode, metric, source, and only the numbers needed by those selections. Unknown
selectors, wrong active types, non-finite active numbers, negative active
color/start/density, or end <= start produce a warning naming the invalid fog
setting and disable the **entire effective fog operation for that commit**.
They are not silently clamped or replaced by the previous effect. Unrelated
Renderer/World settings remain intact; a corrected Commit recovers normally.

## Non-goals

This extension is an artistic, surface-only depth cue. It does not provide:

- Height fog, participating media, or volumetric fog; no atmospheric transport
  through a combined medium.
- Fog on misses (fogged skies), on shadow or secondary segments, or on the
  direct RGB of scientific volumes.
- A new ANARI object, a new Renderer-instance property, a new background form,
  or a KHR extension. No height-fog or volumetric-fog capability is advertised.
- VisGL support.
- Pixel-identical images across Renderer subtypes or against historical
  engines. GL-style exponential start/end are ignored: density 0.1 at distance
  10 gives visibility approximately 0.367879 even when start is 5, not
  0.606531. Converting normalized D3D device-depth endpoints to view distances
  alone does not reproduce an arbitrary depth-buffer fog curve.
- Conversion of lights to Materials; Light Proxies keep their representation.

## Verification

The public-ANARI release tests, tolerances, and the recorded hardware baseline
are indexed in [renderer-fog-conformance.md](renderer-fog-conformance.md). That
baseline is evidence only for the configurations it names.
