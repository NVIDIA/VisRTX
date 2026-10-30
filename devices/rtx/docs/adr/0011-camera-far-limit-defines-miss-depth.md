# A Miss reports the camera's Far Limit as its depth

ANARI 1.1 defines `channel.depth` as Euclidean distance to the camera but never
says what a pixel with no hit reports, and leaves the perspective/orthographic
`near`/`far` clip distances to the renderer when unset. Every RTX renderer now
reports a Miss's depth as that pixel's Far Limit: the distance along the ray to
the `far` clip *plane* (`far / dot(rayDir, cameraDir)` for perspective), the
ray's `tmax` for a ray-buffer camera, and `FLT_MAX` when `far` is unset. This
holds even when a clip plane or invalid ray buffer shortened the ray.

## Considered Options

- **A renderer parameter (`missDepth`)**. Rejected: the camera's `far` already
  is the spec's notion of where a ray ends, so a second knob would duplicate it.
- **Spherical far limit (every Miss reports exactly `far`)**. Rejected: the spec
  calls `far` a clip *plane*, so Miss Depth varies across a perspective image.
- **`+inf` when `far` is unset**. Rejected: `FLT_MAX` is what applications
  already see from the Fast and Interactive renderers and test for, and `inf`
  produces NaN when positions are reconstructed from depth.

## Consequences

- With `far` unset, Miss Depth is the constant `FLT_MAX`, not `FLT_MAX / cos`
  (which would overflow to `inf` toward the image edges).
- Applications should identify a Miss through the ID channels (`~0u`), not by
  comparing depth to a sentinel, once `far` is set.
