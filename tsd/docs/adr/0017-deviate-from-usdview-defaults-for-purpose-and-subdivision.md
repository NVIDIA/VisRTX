# Deviate from usdview defaults for purpose and subdivision

USD import is judged by visual parity with a reference Hydra render, and two
import defaults deliberately differ from stock usdview anyway. usdview defaults
to `showProxy(true)` and `showRender(false)`, so it displays proxy stand-in
geometry; TSD imports `default` + `render` because it renders with a path tracer
for which proxy assets are the wrong input, and because an asset whose real
content sits behind `purpose=render` would otherwise import as its bounding-box
card. usdview defaults to complexity 1.0, which maps to refinement level 0 and
draws subdivision meshes as their unrefined control cage; TSD refines with
OpenSubdiv by default so silhouettes are correct. Both are configurable through
`UsdImportOptions`, and any parity comparison must set matching purpose and
Complexity on the reference render before treating a difference as a defect.
Neither deviation is a bug, and neither should be "aligned" with usdview without
revisiting this decision.
