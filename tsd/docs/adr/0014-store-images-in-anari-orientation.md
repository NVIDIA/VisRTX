# Store images in ANARI orientation

A decoded image resident in a TSD Scene is stored in ANARI orientation: the
array's row 0 is the top row of the picture, because ANARI addresses texture
coordinate `(0, 0)` at the image's upper-left corner. Importers hand ANARI
texture coordinates in ANARI's convention, converting from the source format's
where they differ — glTF's `v` already runs down the image and is passed
through, while OBJ, USD, PBRT, and ASSIMP are all v-up and have their `v`
reversed. A format that also carries a uv transform of its own has it
conjugated by that reversal rather than flipped twice: PBRT's `vscale`/`vdelta`
and USD's `UsdTransform2d` both become `vs*v + (1 - vs - vd)`.

Decoders declare the row order their library produced and `ImageCache`
normalizes; no importer flips texels itself. Previously each of seven decode
paths carried its own unstated assumption, and the assumptions cancelled for
glTF, ASSIMP, and PBRT but not for OBJ and USD, whose textures rendered
mirrored. Two consequences follow from the contract: an image bound somewhere
that is not an image sampler asks `ImageCache` for the order that consumer
wants — an `hdri` light's `radiance` runs bottom-up, so it says so on its
`ImageSource`; and block-compressed DDS, whose 4×4 blocks cannot be
row-reversed without decoding and re-encoding, would stay as authored and
instead get a `v`-flip composed into its sampler's `inTransform`/`inOffset`
— which is why `makeImageSampler` owns those two parameters outright and takes
the importer's own uv transform through `SamplerSettings`. No decoder in the
tree emits bottom-up rows today, so that path is dormant rather than dead.
