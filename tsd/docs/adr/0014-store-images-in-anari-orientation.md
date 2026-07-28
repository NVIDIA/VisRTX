# Store images in ANARI orientation

A decoded image resident in a TSD Scene is stored in ANARI orientation: the
array's row 0 is the bottom row of the picture, so texture coordinate `(0, 0)`
addresses the lower-left corner. Importers hand ANARI texture coordinates in
ANARI's convention, converting from the source format's where they differ —
glTF and PBRT flip `v`, and ASSIMP simply stops asking for `aiProcess_FlipUVs`.
Decoders declare the row order their library produced and `ImageCache`
normalizes; no importer flips texels itself. Previously each of seven decode
paths carried its own unstated assumption, and the assumptions cancelled for
glTF, ASSIMP, and PBRT but not for OBJ and USD, whose textures rendered
mirrored. Two consequences follow from the contract: anything writing a scene
array back out as a top-down image format must reverse its rows, which
`SceneToUSD` now does for PNG and EXR; and block-compressed DDS, whose 4×4
blocks cannot be row-reversed without decoding and re-encoding, stays as
authored and instead gets a `v`-flip composed into its sampler's
`inTransform`/`inOffset` — which is why `makeImageSampler` owns those two
parameters outright and takes the importer's own uv transform through
`SamplerSettings`.
