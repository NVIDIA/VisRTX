# Generated normal-map tangent frames put +Y along +dP/dv

The ANARI spec says the `normal` and `clearcoatNormal` samplers return
tangent-space normals but never fixes which way tangent-space +Y (the
bitangent) points. Where VisRTX generates tangents
(`geometry/ComputeTangent.cu`), the bitangent follows +dP/dv, toward increasing
texture coordinate v, the same as halcyon (anari-halcyon `MeshTangents.cpp`). We chose to
match halcyon so the two ANARI devices agree on meshes without authored
tangents. Authored tangents are unchanged: B = w · cross(N, T), with w taken as
given.

## Considered Options

- **+Y = −dP/dv (up the image, glTF's "+Y is up")**, which VisRTX's generator
  used before by negating the v deltas. Rejected so that VisRTX agrees with
  halcyon.

## Consequences

- With v = 0 at image row 0 (the top of the picture), +dP/dv points down the
  image. A glTF/OpenGL-style normal map on a mesh without authored tangents
  therefore renders with its green channel inverted. Producers that want glTF's
  orientation must author tangents.
- Generated frames and the MikkTSpace tangents Vela authors for glTF
  (`flipTexCoordY`) have opposite bitangents. A mesh can look different
  depending on whether its tangents were authored.
