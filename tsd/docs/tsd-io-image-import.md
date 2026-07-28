# Central image import for tsd_io

> **Status: steps 0-3 landed** (`98e4a1e8`..`21d4d741` on `usd-import-rework`).
> The contract itself is recorded in
> [ADR 0014](adr/0014-store-images-in-anari-orientation.md); this document
> keeps the survey that motivated it and tracks what is left. The Survey below
> describes the tree *before* the change and is retained as the record of why.
> See [Status](#status) and [Remaining work](#remaining-work).

## Summary

Every importer that read texels reached a different decoder, and each one
carried a private, undocumented assumption about which row of the decoded
image is row 0. The assumptions cancelled out for glTF, ASSIMP, and PBRT and
did not cancel for OBJ and USD, which is why textures from those two formats
came out vertically mirrored. Nothing downstream corrected it: no render index
or sampler code in TSD flipped anything.

This proposed one image-import component with a single stated orientation
contract — decoded texels are stored in ANARI orientation, row 0 is the bottom
row — plus a scene-scoped cache so a texture referenced from many places is
decoded once. The contract and the component landed; the cache is still scoped
to one importer call rather than one import.

## Survey

*Everything in this section describes the tree at `017f7d91`, before the
change. It is kept because the reasoning, not just the conclusion, is what a
future decoder author needs.*

### Decode paths in the tree today

| # | Path | Entry point | Decoder | Row 0 |
|---|---|---|---|---|
| 1 | shared, from file | `importTexture` (`detail/importer_common.cpp:526`) | stb / tinyexr / OIIO / DDS by extension | top |
| 2 | shared, from memory | `importTextureFromMemory` (`:564`) | stb or DDS by format hint | top |
| 3 | shared, pre-decoded | `importRawTexture2D` (`:593`) | `memcpy` of RGBA8 | caller's |
| 4 | glTF | `importGLTFTexture` (`import_GLTF.cpp:104`) | tinygltf's own decode | top |
| 5 | PBRT height→normal | `importHeightAsNormalMap` (`import_PBRT.cpp:1212`) | private `stbi_loadf` | top |
| 6 | HDRI environment | `HDRImage::import` (`detail/HDRImage.cpp:171`) | stb with `stbi_set_flip_vertically_on_load(1)`, plus a hand-written flip in the EXR branch (`:81`) | **bottom** |
| 7 | PBRT equirect resample | `import_PBRT.cpp:125` | consumes #6's buffer | **bottom** |

Path 6 is the only producer that flips. Its comment at `HDRImage.cpp:173`
("Restore default top-down orientation") is the only place in the tree that
names an orientation at all, and it names the *opposite* of what paths 1–5
produce.

Paths 4 and 5 duplicate cache-key construction, `Array` creation, and sampler
construction that paths 1–3 already have. Path 4 additionally does something
paths 1–3 do *not*: it preserves the file's integer element type and uses
ANARI's `*_SRGB` formats, where the shared path expands everything to
`ANARI_FLOAT32*` and applies `pow(x, 2.2)` in software.

`importGLTFTexture` also takes a `flipNormalMapY` parameter that is only ever
folded into the cache key (`import_GLTF.cpp:134`) and never applied to the
texels; no caller passes it.

### Orientation, end to end

Sampling comes out right when the row order of the stored array and the `v`
convention of the texture coordinates agree. Today they agree by accident in
three importers and disagree in two:

| Importer | Source `v` convention | UV handling at import | Array row 0 | Result |
|---|---|---|---|---|
| glTF | v-down (spec) | passed through (`import_GLTF.cpp:1053`) | top | correct |
| ASSIMP | v-up (assimp default) | `aiProcess_FlipUVs` → v-down (`import_ASSIMP.cpp:695`) | top | correct |
| PBRT | v-up | `v = 1 - v` (`import_PBRT.cpp:221`, `:429`) → v-down | top | correct |
| OBJ | v-up (`vt` spec) | passed through (`import_OBJ.cpp:132`) | top | **mirrored** |
| USD | v-up (UsdPreviewSurface / MaterialX `st`) | passed through | top | **mirrored** |

So the working importers are all on the GL contract — top-down image, v-down
coordinates — and the two that hand ANARI genuinely v-up coordinates are
broken. `import_PBRT.cpp:909` states this explicitly and calls the v-down
convention "ANARI's", which is backwards; the same comment then documents the
`(1 - vs - vd)` term in `applyPbrtUvTransform` as compensation for the double
flip.

Two further consequences of storing every array upside down relative to
ANARI's definition:

- Anything that reads a scene's texture array back *as an image* sees it
  flipped. `SceneToUSD.cpp:167`/`:195` writes arrays straight to EXR and PNG,
  both of which are top-down formats — correct today only because the arrays
  are top-down too.
- `calcTangentsForTriangleMesh`'s `flipTexCoordY` parameter
  (`importer_common.cpp:641`, defaulting to `true`) exists solely to undo the
  v-down convention before handing coordinates to mikktspace.

### Cache

`TextureCache` is `unordered_map<std::string, ArrayRef>`
(`importer_common.hpp:31`), keyed by `path + "_linear"|"_srgb"`
(`makeTextureCacheKey`). Every importer constructs its own and drops it when
the import returns:

- `import_OBJ.cpp:59`, `import_ASSIMP.cpp:246`, `import_GLTF.cpp:307`,
  `import_PBRT.cpp:2321` — function-local
- `UsdImportContext.h:56` — per-import context
- `import_HDRI` — none at all

Reuse therefore exists *within* one importer call and nowhere else. Importing
two assets that share a texture decodes it twice; a USD stage that references
an OBJ decodes it twice.

The cache also has no tie to the `Scene` whose arrays it holds. Nothing
structurally prevents handing scene A's `ArrayRef` to scene B, or outliving
the scene entirely.

Key construction is inconsistent: paths 1–3 use `makeTextureCacheKey`, glTF
hand-rolls `name + "_srgb" + "_yflip"` (`import_GLTF.cpp:128-137`), and PBRT's
height map uses `path + "::normal"` (`import_PBRT.cpp:1219`).

## Proposal

### The contract

> A decoded image resident in a TSD scene is stored in ANARI orientation: the
> array's row 0 is the bottom row of the picture, so texture coordinate
> `(0, 0)` addresses the image's lower-left corner. Importers hand ANARI
> texture coordinates in ANARI's convention, converting from the source
> format's convention where they differ.

Decoders declare the row order their library produces; the import layer
normalizes. No importer flips anything itself.

### Component

A new `src/tsd/io/images/` alongside `importers/`, since this is shared by
importers and exporters both:

```cpp
namespace tsd::io {

enum class ColorSpace { SRGB, LINEAR };

// The row order a decoder produced. Declared by decoders, never by importers.
enum class RowOrder { TOP_DOWN, BOTTOM_UP };

// Identifies texel content — not the sampler built from it. Two materials
// binding the same file at the same color space share one Image.
struct ImageSource
{
  std::string id;           // resolved path, or an importer-scoped stable id
  std::string displayName;  // sampler name; defaults to fileOf(id)
  ColorSpace colorSpace = ColorSpace::SRGB;
};

// A decoded image resident in a Scene, in ANARI orientation.
struct Image
{
  tsd::scene::ArrayRef texels;
  bool blockCompressed = false;
  explicit operator bool() const { return texels.valid(); }
};

// Owns decoded images for one Scene. Holds the Scene it caches for so a
// cached ArrayRef can never reach a different Scene; it must not outlive
// that Scene. Follows the `Scene *m_scene{nullptr}` member convention used
// by Layer, AnariHandleCache, and the network messages.
class ImageCache
{
 public:
  ImageCache(tsd::scene::Scene *scene);

  tsd::scene::Scene *scene() const;

  Image acquire(const ImageSource &source);
  Image acquire(const ImageSource &source,
      const void *data,
      size_t numBytes,
      const std::string &formatHint = "");
  Image acquireDecoded(const ImageSource &source,
      anari::DataType elementType,
      size_t width,
      size_t height,
      RowOrder rowOrder,
      const void *texels);

  void clear();
  size_t size() const;

 private:
  tsd::scene::Scene *m_scene{nullptr};
  // ...
};

struct SamplerSettings
{
  const char *inAttribute = "attribute0";
  const char *wrapMode1 = "repeat";
  const char *wrapMode2 = "repeat";
  const char *filter = "linear";
};

tsd::scene::SamplerRef makeImageSampler(tsd::scene::Scene &scene,
    const Image &image,
    const std::string &displayName,
    const SamplerSettings &settings = {});

} // namespace tsd::io
```

Caching stays at the array level, as it is today: samplers are cheap and their
wrap/filter/`inAttribute`/`inTransform` differ per binding, so they are built
fresh. `Image` is the unit of sharing.

`acquireDecoded` is what lets glTF and any future format that arrives
pre-decoded (tinygltf, an embedded DDS, a procedural buffer) join the shared
path — it declares its row order and gets the same normalization, keying, and
lifetime as a file-backed image.

`importTexture` / `importTextureFromMemory` / `importRawTexture2D` survive as
thin wrappers so the ~20 call sites don't churn in the same commit as the
behavior change.

### Where the flip happens

One place: `ImageCache`'s store step, between decode and `Array::setData`. Each
decoder reports `RowOrder`; stb, tinyexr, OIIO, and tinygltf report
`TOP_DOWN`, and the cache reverses rows before the texels reach the scene.

> **As landed:** `HDRImage` did *not* stop flipping. It keeps its own decode,
> because it handles multipart EXR and forces three channels and the shared
> path does neither, and it declares `BOTTOM_UP` rather than flipping twice.
> That satisfies the contract — a decoder declares, the cache normalizes —
> but it does mean two places in the tree reverse rows. See
> [Remaining work](#remaining-work).

**Block-compressed DDS is the one exception.** BC blocks are 4×4, so a
vertical flip requires decode and re-encode, which defeats the point of
`compressedImage2D`. Recommendation: keep DDS texels as authored and mark the
`Image` so `makeImageSampler` folds a `v`-flip into that sampler's
`inTransform`/`inOffset` (`diag(1, -1, 1, 1)`, offset `(0, 1, 0, 0)`). This is
exact and costs nothing at runtime. Callers that set their own `inTransform`
(USD's `uvTransform`, PBRT's `uscale`/`vscale`, glTF's `KHR_texture_transform`)
must compose rather than overwrite — a `composeVFlip(mat4 &, float4 &)` helper
keeps that honest. The alternative, decoding DDS to RGBA and flipping, is
simpler but throws away the compression.

> **As landed:** this recommendation was ratified and implemented, with one
> change of shape. Rather than a `composeVFlip` helper that callers must
> remember to use, `makeImageSampler` owns `inTransform`/`inOffset` outright
> and takes the importer's own transform through `SamplerSettings`. A caller
> cannot overwrite the flip, because it no longer sets those parameters
> itself. `tests/test_ImageImport.cpp` asserts both the flip and that it
> composes onto a caller's transform rather than replacing it.

### Importer changes that must land with the flip

Flipping the arrays without these is a regression, so they belong in one
commit:

| Importer | Change |
|---|---|
| glTF | flip `v` when building `vertex.attributeN` (`import_GLTF.cpp:1053`), since glTF is v-down |
| ASSIMP | drop `aiProcess_FlipUVs` (`import_ASSIMP.cpp:695`); assimp's default output is already v-up |
| PBRT | drop `v = 1 - v` at `:221` and `:429`; simplify `applyPbrtUvTransform` (`:913`) to plain `(us*u + ud, vs*v + vd)` and delete the compensating comment at `:906` |
| OBJ | none — becomes correct |
| USD | none — becomes correct |
| `SceneToUSD` | flip rows when writing PNG (`:195`) and EXR (`:167`), which are top-down formats |
| `calcTangentsForTriangleMesh` | default `flipTexCoordY` to `false`; both callers (`import_ASSIMP.cpp:217`, `import_GLTF.cpp:1213`) pass `false` |

One wrinkle this table missed: glTF's tangent path does not read the texture
coordinates it just stored, it re-reads the raw accessor data, which is still
glTF's v-down. So it flips that local copy too before handing it to
mikktspace. ASSIMP's caller reads the stored coordinates and simply takes the
new default.

Normal maps are unaffected. mikktspace is fed v-up coordinates in both the old
and new schemes — today via `flipTexCoordY = true` undoing the v-down
convention, afterward because the coordinates are already v-up — so the tangent
basis is invariant, and the fetch coordinates and row order change together.

### Cache lifetime

`ImageCache` is a value type the caller owns, scoped to the `Scene` it points
at:

- `import_file()` creates one per call and threads it to whichever importer it
  dispatches to. That is the "temporary" scope requested: a scene referencing
  the same texture from many materials, or from a nested asset in another
  format, decodes it once.
- An overload taking an existing `ImageCache &` lets an application that
  imports many files as one operation (a SciVis Studio project load) own one
  cache across all of them.
- Nothing caches across unrelated user actions, so a texture edited on disk is
  picked up on the next import without invalidation machinery.

> **Not implemented.** `ImageCache` exists and is scoped to a `Scene`, but each
> importer still constructs its own, so reuse stops at the importer-call
> boundary. This is [remaining work item 1](#1-one-cache-per-import-not-per-importer).

`ImageSource::id` is the resolved absolute path for file-backed images and an
importer-scoped stable string otherwise (`"gltf:<file>:image<N>"`,
`"assimp:<file>:embedded<N>"`, `"pbrt:<file>::normal"`). The cache key is
`(id, colorSpace)`, replacing the three key-construction schemes in the tree.

### Follow-on: preserve element types

The glTF path already keeps the file's integer type and uses ANARI's `*_SRGB`
element formats; the shared path expands to `ANARI_FLOAT32*` and applies
`pow(x, 2.2)` in software (`images/detail/decoders.cpp:251`). Moving the
shared path onto native types would cut texture memory 4× for the common
8-bit case and replace the 2.2 gamma approximation with the true sRGB EOTF the
device applies, deleting `applyGamma22InPlace` and the comment above it
explaining why the OIIO path has to imitate stb's approximation. This is worth
doing but is a separable change; it should not ride along with the orientation
fix.

## Status

| Step | | |
|---|---|---|
| 0 | Characterize | **Landed** `98e4a1e8` |
| 1 | Introduce `tsd/io/images` | **Landed** `d27547b8` |
| 2 | Flip | **Landed** `c4106e72` |
| 3 | Fold in the stragglers | **Landed in part** `860cf337` |
| 4 | Native element types | **Not started** |

Step 0 became `tests/test_ImageImport.cpp` (tag `[ImageImport]`). It ran red
exactly where this document predicted — glTF and PBRT passed the end-to-end
assertion, OBJ and USD failed it — which is the local evidence for the survey
above, arrived at independently of the source reading that produced it.

Two corrections to this document's step 0, established before it was written:
fixtures are synthesized into the temp directory rather than checked in,
following the TGA in `tests/test_UsdImport.cpp` and the TIFF in
`tests/test_Importers.cpp`; and the assertion is a scene query rather than a
render, so no device is needed. One 1x2 TGA serves OBJ, glTF, PBRT, USD, and
ASSIMP; DDS and Radiance HDR have fixtures of their own.

## Remaining work

Roughly in the order that pays off soonest.

### 1. One cache per import, not per importer

This document's [Cache lifetime](#cache-lifetime) section is the one part of
step 1 that was not implemented. Every importer still builds a function-local
`ImageCache` (`import_OBJ.cpp:59`, `import_GLTF.cpp:259`,
`import_ASSIMP.cpp:249`, `import_PBRT.cpp:2321`, `UsdImportContext.h:56`,
`import_HDRI.cpp:33`), so reuse still exists only *within* one importer call.
The stated payoff — "a USD stage that references an OBJ decodes it twice" —
is unfixed.

What it needs: `import_file()` owning one cache and threading it to whichever
importer it dispatches to, plus the overload taking an existing `ImageCache &`
for an application importing many files as one operation. That is an
`ImageCache &` parameter across the importer signatures in `importers.hpp`,
which is why it did not ride along with a behavior change.

Prerequisite already done: `ImageSource` ids are file-scoped
(`gltf:<file>:<image>`), so sharing a cache across files is safe.

### 2. Retire the shims

`importTexture`, `importTextureFromMemory`, and `importRawTexture2D`
(`importer_common.hpp:32-54`) are pure forwarding to `ImageCache` and
`makeImageSampler`. They exist so the ~20 call sites did not churn in the same
commit as the behavior change, which has now happened. Two ways to do the same
thing is the state this work set out to remove, so they should go once
something else is touching those call sites anyway — item 1 is the natural
moment.

### 3. PBRT's infinite light

`loadInfiniteRadiance` (`import_PBRT.cpp:2094`, `:2107`) is the last place
outside `ImageCache` calling `scene.createArray` for image data. It consumes
`HDRImage`'s raw buffer and, for a square source, runs it through
`convertEqualAreaToEquirectangular`, whose frame of reference is documented in
terms of the buffer layout it is handed.

It is correct as it stands. Moving it onto the cache means rewriting that
conversion's frame of reference — three sign changes — and there is no test
holding it, because a PBRT equal-area HDRI is not cheap to synthesize. Write
the fixture first.

### 4. `HDRImage`'s own flip

`HDRImage` reverses rows in both branches (`HDRImage.cpp:81`, `:173`) and
declares `BOTTOM_UP`. The contract holds, but "one place flips" does not.
Folding it into the shared path means teaching `decodeImageFile` about
multipart EXR and about forcing three channels; worth doing when something
else needs multipart EXR, not before.

### 5. Native element types

Unchanged from [the follow-on above](#follow-on-preserve-element-types), and
still the largest single win: the shared path expands every image to
`ANARI_FLOAT32_*` (`decoders.cpp:43`) and applies `pow(x, 2.2)` in software
(`decoders.cpp:251`). Moving to the file's own type and ANARI's `*_SRGB`
formats — which the glTF path already does — would cut texture memory 4× for
the common 8-bit case and replace the gamma approximation with the true sRGB
EOTF the device applies, deleting `applyGamma22InPlace` and the comment above
it explaining why the OIIO path has to imitate stb.

Note this interacts with the orientation tests: they read texels through a
helper that already handles both `ANARI_FLOAT32_*` and `ANARI_UFIXED8_*`, so
they should survive the change unmodified. That is deliberate.

### 6. Test coverage gaps

- `decodeExr` and `decodeOiio` both declare `RowOrder::TOP_DOWN` with nothing
  asserting it. An EXR fixture is cheap — tinyexr can write one. A TIFF
  fixture exists in `tests/test_Importers.cpp` but is 1x1, so it says nothing
  about row order; widening it to 1x2 would.
- `convertEqualAreaToEquirectangular` is untested, which is what blocks item 3.

## Discovered along the way

Not part of this work, recorded because the tests surfaced it and it will
mislead someone otherwise.

**ASSIMP binds no textures for OBJ.** ASSIMP reports a GL-style shading model
for OBJ files, and that branch of `importASSIMPMaterials`
(`import_ASSIMP.cpp`, the `else // GL-like dflt. material` case) sets colour
and opacity and nothing else — no texture slot is read. An OBJ with a
`map_Kd` therefore imports untextured through ASSIMP, while the same file
imports correctly through `import_OBJ`. This is why the ASSIMP orientation
test goes through the glTF fixture instead.
