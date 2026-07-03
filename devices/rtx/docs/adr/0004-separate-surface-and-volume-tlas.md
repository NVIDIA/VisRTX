# Separate surface and volume TLAS

The world builds two top-level acceleration structures — one for surfaces, one
for volumes — traversed independently, instead of a single TLAS with OptiX
visibility masks. The reason is hit semantics, not performance: surface
traversal yields hit points, while volume integration needs to enumerate
entry/exit intervals along the ray, decoupled from surface hits. Separate
traversables keep the hit programs and SBT for each simple. Visibility masks
were not seriously considered.

## Consequences

- Rays that need both surfaces and volumes trace twice.
- Surface and volume acceleration structures rebuild independently.
