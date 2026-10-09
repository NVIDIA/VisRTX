# RTX fog release conformance

The public interface under test is ANARI: capability/parameter queries, object
creation and Commit, render/wait, mapped channels, and status callbacks. Tests
live in [`../apps/tests/api/`](../apps/tests/api/) and are individually registered
with CTest. All six image executables require discovery on all four supported
names; absence is a failure, not a reason to skip rendering.

In the table, **all** means `fast`, `interactive`, `default`, and `quality`.
Quality's stochastic coverage/volume cases use their own references, not a
straight-through Renderer as ground truth. Function names and quoted case labels
identify executable cases; no private fog evaluator supplies expected values.

## Required acceptance matrix

| # | Requirement | Executable / named cases | Scope and reference |
|---|---|---|---|
| 1 | Discovery, aliases, defaults, exclusions | `TestRendererFogDiscovery::testDiscovery`; `TestRendererFog::testDiagnosticRenderers` | All seven parameters, types/defaults/selectors, schema identity, descriptions/units/applicability, device query/property and utility, every enumerated subtype; debug/test/debug_Ng/unknown exclusion, no other fog strings; diagnostic RGB unchanged |
| 2 | Omitted/none/inactive-invalid output | `TestRendererFog::testColor`; `TestRendererFogCoverage::testDisabledChannels` | All; bit-identical known-color output and paired 64-spp layered RGBA/depth/IDs/normal/albedo, including inactive wrong types/selectors/numbers |
| 3 | Linear endpoints and short interval | `TestRendererFog`: `linear endpoints/clamping`, `short linear interval` | All; before/start/mid/end/beyond and interval `[1e-8,3e-8]` |
| 4 | Exponential modes and ignored endpoints | `TestRendererFog`: `exp`, `exp2`, `zero distance`, `zero distance exp2`, `exponential ignores changed start/end` | All; independent long-double optical-distance equations; zero density, zero distance, moderate and high attenuation |
| 5 | Unshifted legacy exponential | `TestRendererFog::testColor`: `unshifted exp visibility, not 0.606531`; `legacy unshifted exp regression` | All; density .1, distance 10, start 5 gives visibility .36787944 |
| 6 | Monotonicity, bounds, finite extremes | `TestRendererFog::testDistanceSweep`, `testColor` | All; 24 ordered distance references per subtype; subnormal density/HDR, tiny squared argument, maximum finite start/end/color/density. Quality HDR surface-radiance attenuation in `TestRendererFogTransport::testHdrRadiance` |
| 7 | Perspective versus orthographic distances | `TestRendererFogDistance::testPlanes` | All; both metrics and all active curves against geometric pixel-center references |
| 8 | Translation and scale covariance | `TestRendererFogDistance::testPlanes` | All; joint translation `(13,-7,31)`, scales .125/1/8, scaled distances or inverse-scaled density |
| 9 | Ray buffers, clipping, pass-through, aperture | `TestRendererFogDistance::testRayBuffer`, `testOpaqueCutPlane`, `testAperture`; `TestRendererFogCoverage::testLayers` | All; supplied origins/directions/intervals, camera reference edits, cutting-plane re-origining, independent two-layer depths; Quality coverage in `testLayers`; analytic lens integral |
| 10 | Constant/HDR RGB and output encoding | `TestRendererFog::testColor`; `TestRendererFogBackground::testEncoding` | All; linear float equations; independent linear-to-sRGB encoding, one 8-bit code-value allowance |
| 11 | Supported backdrop forms | `TestRendererFogBackground::testFlat`, `testImage`, `testHdri`, `testDirectionalHdri` | All under both metrics; independent image interpolation, image region/off-center samples, HDR, visible/invisible/multiple HDRIs, precedence, orientation/instance transforms, premultiplication |
| 12 | Miss/alpha preservation; ignored constant tint | `TestRendererFogBackground::testFlat`, `testImage`, `testHdri`, `testLifecycle`; `TestRendererFogInteractions::testStraightThroughMatrix`, `testQualityProxyCoverage` | All; opaque/transparent backdrops, unchanged misses, invalid/wrong-type fogColor ignored in background mode |
| 13 | Distinct layers, masks, transparency | `TestRendererFogCoverage::testLayers`, `testCutouts`, `testTransmission` | All; both metrics, constant/background layers, independent premultiplied formula. Native straight-through colored transmission on fast/interactive/default; Quality reflection/refraction in `TestRendererFogTransport::testTransport` |
| 14 | Per-sample silhouette/depth edges | `TestRendererFogCoverage::testEdges` | All; 256 paired spp, independently pre-cued contribution colors; a wrong resolved-depth postprocess must disagree on at least eight mixed pixels |
| 15 | Once-only secondary transport | `TestRendererFogTransport::testTransport`; `TestRendererFogBackground::testQualityIndirect` | Quality; 4096 spp reflection/refraction, controlled-radiance indirect emitters moved between two depths, secondary proxies, four-interface RR, both metrics and constant/image sources |
| 16 | Light Proxies and illumination | `TestRendererFogInteractions::testProxy`, `testIllumination`, `testStraightThroughMatrix`, `testQualityProxyCoverage`; `TestRendererFogTransport::testTransport`; `TestRendererFogCoverage::testLighting` | All existing visibility policies; independent radiance/opaque references, hidden/unpickable proxies, unaffected illuminated receiver; Fast's exclusion is asserted, not added as a feature. `testLighting` is the AO-only control: fog preserves the ambient-occlusion contribution and direct lighting on fast/interactive/default and quality |
| 17 | Volume-only and surfaces behind volumes | `TestRendererFogInteractions::testVolume`, `testJitteredVolume`, `testStraightThroughMatrix`, `testQualityVolume`; `TestRendererFogTransport::testIndirectVolume` | All; analytic emission-absorption or Beer survival, direct volume invariance, absorbing/scattering Quality cases, surface/proxy depth, no new fog after volume scattering |
| 18 | Auxiliary meanings, full fog, cutouts | `RendererFogSupport.h::sameAuxiliary` (paired exact comparison used by `TestRendererFogCoverage`, `TestRendererFogDistance`, `TestRendererFogInteractions`, `TestRendererFogTransport`, `TestRendererFogBackground::testQualityIndirect`); `TestRendererFogCoverage::testDisabledChannels` | All; paired exact alpha/depth/primitive/object/instance IDs/normal/albedo across partial/full/none fog, live metadata/lighting controls |
| 19 | Recommit, unset, recovery, accumulation | `TestRendererFog::testLifecycle`; `TestRendererFogDistance::testMetricLifecycle`; `TestRendererFogBackground::testLifecycle`; `TestRendererFogCoverage::testLayeredRecommit` | All; each active parameter, fresh/reused equivalence, mode/source/metric changes after four old accumulation frames, none/unset defaults |
| 20 | Background invalidation and shared Worlds | `TestRendererFogBackground::testImage`, `testDirectionalHdri`, `testLifecycle`, `testResourceUnwinding`; `TestRendererFog::testLifecycle` | All; both metrics, flat/image/HDRI observed edits, two live Frames/Renderers sharing a World with different settings; an array edit whose map/unmap scope unwinds through an exception still publishes and invalidates dependent color, and interrupted frame mappings leave later renders intact |
| 21 | Invalid active values/types and recovery | `TestRendererFog`: `invalid` case driver and `corrected commit`; `TestRendererFogDistance::testMetricLifecycle`; `TestRendererFogBackground::testLifecycle` | All; relevant warnings, unknown strings/wrong types, NaN/infinity/negatives, bad interval, effective-none output, preserved backdrop and fresh/reused recovery |
| 22 | Existing disabled regressions | `TestRayBufferCamera`, `TestAlphaCutoffMode`, `TestSamplerAlphaFill`, `TestConeCylinderTransparency`, `TestFrameChannelTypes`, `TestVisibleAreaLight`, `TestEmissiveRecommit` | Existing camera/clipping/ID, alpha/transparency, channels, Light Proxy and Commit checks; full configured RTX suite also runs |

## Tolerances and non-vacuous controls

- Deterministic linear float RGB: `abs(error) <= 1e-5 + 1e-4*abs(reference)`.
  Exact paired equality is used for unchanged channels/disabled output. Encoded
  sRGB comparisons independently encode linear RGB and allow one code value.
- Quality HDR attenuation: one paired sample avoids overflowing the unfogged
  accumulation sum. Direct emission and reflection exercise `exp`/`exp2`
  visibility below the FLOAT32 normal range against long-double affine
  references, with the deterministic RGB tolerance and exact auxiliary checks.
- AO-only control: fast/interactive/default use 64 spp × 16 AO samples, no
  direct irradiance, and radii .25/20. The rear receiver must recover .12
  unoccluded radiance at the short radius and darken by more than .02 at the
  long radius; a paired linear-fog render must preserve that AO contribution.
- Lens: 4096 spp over 16 pixels (65536 samples), aperture radius 8 and focal
  distance 10. The independent radial density is `2*d/64` on
  `[10,sqrt(164)]`; limits are .007 per pixel and .002 image mean. These are
  conservative six-standard-error criteria, not an IID claim for Halton samples.
- Quality layers: 4096 spp, .05 per-pixel RGBA and .008 mean error in each of six
  disjoint origin/half-image groups (at least 40 pixels/group). Zero-opacity
  and cutout cases are included. Edge references use paired 256-spp sampling
  and retain the deterministic combined tolerance.
- Quality transport: 4096 spp; opaque once-only affine comparisons retain the
  deterministic tolerance. Partial front coverage allows .05 pixel / .009
  per-origin-group error. Controlled emitter/tint means and a max-depth-one
  control establish actual secondary radiance. The four-interface RR mean has
  a .025 bound; fog may not alter paired transport/AOVs.
- Straight-through jittered slabs: 16384 samples, transmission allowance
  .000329618705 from independently bounded quadrature bias plus Hoeffding error
  (failure probability 1e-9), scaled by color contrast plus numeric tolerance.
  Integral-step slabs and the oblique cross-product are deterministic.
- Quality camera survival/coverage: 4096 spp, two groups of 120 pixels;
  Hoeffding bounds .0511303 per pixel / .00466754 per group for unit-range
  observations (two-sided failure probability 1e-9 per assertion), plus numeric
  tolerance. Paired scattering-volume lighting cancels in the surface-difference
  oracle. Volume-only RGB and all auxiliary channels remain paired-identical.

The release cross-product explicitly runs background-source layers and layered
state transitions on all four names. Fast/interactive/default additionally run
120 interaction cases each: two metrics × two sources × five modes
(none/linear/exp/exp2/full) × three contribution types (backdrop/surface/proxy) ×
volume present/absent. Fast's absent proxy is an expected unchanged backdrop,
not a skipped case. Quality's corresponding absorbing/scattering/coverage
matrix and indirect volume/proxy references are separate cases above. All
background forms and background lifecycle cases run under both metrics.

## Reproduce the baseline

Configure a fresh build (reuse it for subsequent runs):

```sh
cmake -S . -B _out/fog -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON \
  -DOPTIX_FETCH_VERSION=9.0 -DVISRTX_MIN_ARCH=89 \
  -DVISRTX_ENABLE_MDL_SUPPORT=OFF -DVISRTX_ENABLE_MATERIALX_SUPPORT=OFF \
  -DVISRTX_LIBRARY_NAME=visrtx_fog
cmake --build _out/fog --parallel 4
LD_LIBRARY_PATH="$PWD/_out/fog:$LD_LIBRARY_PATH" \
  ctest --test-dir _out/fog -C Release --output-on-failure
```

A focused run uses `-R '^TestRendererFog'`; add `-V` to see the quantitative
measurements. Prepending the build's library directory is important if another
VisRTX is installed. Check `ldd _out/fog/TestRendererFogDiscovery` under the same
environment before interpreting a run as release evidence.

### Recorded release baseline (2026-10-08)

The verified baseline uses native materials and structuredRegular volumes, an
RTX 5880 Ada GPU, CUDA 13.2, OptiX 9.0, and a Release RTX-only build with
MDL/MaterialX disabled. It is not evidence for untested material
configurations, other GPUs, or GL.

**All 22 rows passed** in the native-material RTX-only configuration on NVIDIA
RTX 5880 Ada Generation, driver 615.71.09, CUDA 13.2.51, OptiX 9.0.
The final full run passed **47/47 CTests, no skipped test or promised Renderer
subtype**, in 194.73 seconds. This includes all seven fog executables and the
existing disabled-mode regressions. MDL/MaterialX configurations, other GPUs,
and other backends were not tested or claimed by this baseline.

Representative measurements from that run:

- New straight-through interaction matrix: maximum absolute error
  `3.72529e-7` on each of fast/interactive/default (120 cases each).
- Omitted/none/inactive-invalid layered channels: zero mismatches on all four
  names at paired 64 spp. Layered recommits matched fresh Renderers after four
  old accumulation frames (1 spp straight-through, 4096 spp Quality).
- Lens means, all names: linear `.57304336` vs `.57303372`; exp `.43573207` vs
  `.43572680`; exp2 `.28031677` vs `.28030917`.
- Quality absorbing-volume surface/proxy maximum group errors `.000326090` /
  `.000265055`; scattering-volume surface/proxy `.00108180` / `.000442176`.
  Indirect volume paired maximum RGB error `.0000337958`.
- Controlled indirect reflection/refraction mean red `.960205` / `.959974`
  versus `.96`, unchanged when the emitter moves. The refracted-volume
  fixture's indirect red `.0640569` proves live secondary volume radiance.
- Existing visible-light regression: floor/proxy oracle relative error
  `.001273`, reflected-light oracle `.001732`, occlusion parity `.000432`.
  Existing alpha, ray-buffer, frame-channel, and recommit checks passed.

Discovery was red before publication (missing capability and metadata), then
green after schema generation. Utility discovery independently failed six
support checks before wiring the new field. A temporary mutation making
background-source `rayDistance` use view depth failed the interaction and
coverage suites, including the newly combined cases; it was removed before
the final build. No production evaluator, integrator, transport, or tolerance
change was needed to close this release verification ticket.
