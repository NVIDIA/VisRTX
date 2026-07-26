# Report UDIM tile sets as unsupported rather than approximate them

A MaterialX `filename` input whose path contains a `<UDIM>` marker names a set
of tiles, not a file. `UsdMaterials.cpp` reports such an input as
`TEXTURE_LOAD_FAILED` with `(tiled texture sets are not supported)` and binds no
sampler, while still writing the anchored absolute path into the generated
document. On the OpenPBR Shader Playground reference asset that accounts for 89
of the importer's 100 skipped prims -- every remaining texture failure after
TIFF decoding landed.

Two routes were investigated and both are closed at the level TSD can reach.

The device cannot resolve the tile set on TSD's behalf. VisRTX does load MDL
texture resources from disk -- `SamplerRegistry::loadFromImage` reads
`textureDesc.url` with stb -- but `libmdl::Core::resolveResource` returns
`get_element(0)->get_filename(0)`, a single filename. MDL's entity resolver
returns one element per tile for a `<UDIM>` resource; VisRTX discards all but
the first, and `loadFromImage` then builds exactly one `Image2D`. Nothing in
`devices/` carries a UDIM concept. So MDL's native `<UDIM>` support, which is
real, is not plumbed through this device at all, and setting
`mdlResourceSearchPaths` -- the seam that would fix the cosmetic
`Failed to resolve texture resource` log noise -- would not change the outcome.
Enabling it properly means multi-tile resolution in libmdl, a tile-indexed
texture in the MDL runtime PTX, and a representation for tiled textures in
ANARI, which has none: an ANARI sampler is a single image.

TSD cannot expand the tiles either, for the same reason. Rewriting the
MaterialX network into per-tile branches is a large amount of machinery, and
the obvious shortcut -- binding tile 1001 and dropping the rest -- is wrong for
every mesh that actually spans more than one tile, and worse, it converts a
reported gap into a silently incorrect render.

The gap is therefore left reported rather than approximated. Two properties
make that a deliberate stance and not neglect: the skip is counted in the
import report with a reason, so the cost is visible, and the anchored absolute
path is still written into the document, so a consumer that gains UDIM support
finds a well-formed path waiting. `SdfAssetPath::GetResolvedPath()` is empty
for every UDIM path -- a `<UDIM>` path names no file, so no resolver resolves
it -- which is why that path comes from `UsdMaterials.cpp`'s Stage-directory
fallback anchor rather than from USD's own resolution.

Revisit this when ANARI gains a tiled-texture or texture-array sampler, or when
VisRTX's MDL runtime grows tile-indexed lookup. Until one of those exists,
"supporting UDIM in TSD" has nowhere to send the texels.
