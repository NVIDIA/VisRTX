# Geometry Lights share the light sampling path

Emissive surfaces are turned into Geometry Lights: internal (never
ANARI-creatable) `Light` host objects, one per Emissive Surface, occupying
regular `registry.lights` slots and instanced as `{lightIndex, xfm}` like any
authored light. Unification with existing lights happens at the pick /
sample / MIS (multiple importance sampling) level only — the analytic lights
(Rect, Ring, Sphere, …) keep their closed-form samplers. Reimplementing
those analytic lights as emissive geometry internally was rejected: it would
churn stable code for no rendering gain and foreclose upgrading them to
solid-angle samplers (e.g. cone sampling for spheres), which generic area
sampling cannot match. The inverse — mapping emissive geometry onto analytic
light types — is a dead end: arbitrary meshes and cones have no analytic
equivalent.

## Consequences

- Anything that iterates `lightInstances` (the light pick, interactive's
  all-lights loop, volume light sampling) gains emissive-surface support
  without per-renderer sampling code.
- Geometry Lights are the second light type reachable by BSDF rays, after
  the environment. Every renderer that deposits path-hit emission must
  generalize its environment-only MIS weighting, or next-event estimation
  plus the hit deposit double-counts the emitter. The hit-side weight
  requires a reverse lookup from the hit surface to its light instance.
- `{lightIndex, xfm}` cannot represent emission driven by instance-uniform
  attributes. Once non-constant emission becomes sampleable, the light
  instance must also reference its surface instance so next-event radiance
  matches path-hit radiance.
- A Geometry Light needs a host object to own a registry slot; that object
  is owned by its `Surface` and is not part of the ANARI object model. Do
  not "fix" `Light::createInstance` to expose it.
- The per-primitive area CDF lives on `Geometry` (shared across surfaces,
  rebuilt with the BLAS), not on the light.
