# MaterialX distribution documents resolve at runtime

The device never embeds or ships MaterialX distribution content. The standard
library (nodedefs and their MDL implementation modules) is resolved from a
MaterialX installation at **runtime**, through one chain — first hit wins:

1. `materialxSearchPaths` device parameter — the application's explicit choice,
   same idiom as `mdlSearchPaths` (colon-separated, re-synced on device commit).
2. `MATERIALX_SEARCH_PATH` environment variable — the MaterialX ecosystem
   convention, zero-code for users.
3. `mx::getDefaultDataSearchPath()` — MaterialX's own runtime discovery,
   relative to the loaded MaterialX library; a properly installed MaterialX
   "just works".
4. The compile-time `MATERIALX_LIBRARIES_DIR` bake — a development-build last
   resort only (static-linked MaterialX in a build tree, where (3) resolves
   relative to the device `.so` and misses). It is never the mechanism.

The MDL modules directory is derived as `<resolved root>/libraries/mdl` from
whichever root wins (the root is the directory *containing* `libraries/`); it
is not a separately baked path.

Consequently the `visrtx::standard_surface` builtin alias, the shipped
`standard_surface.mtlx` instantiation file, its install rule, and the
`VISRTX_MATERIALX_STD_SURFACE_MTLX` bake are removed. An application wanting a
default standard_surface authors the instantiation itself: a few lines of
generated `.mtlx` XML passed as `sourceType="documentInline"`. The nodedef the
instantiation binds comes from the distribution via the chain above. An
instantiation is *scene content*, not distribution content — generating it in
the application is not embedding.

Applications are thin conduits: they forward a user-supplied path into
`materialxSearchPaths` and otherwise stay silent. Discovery lives in exactly
one place — the device, which links MaterialX and can ask it. (TSD follows
this: its StandardSurface preset emits the inline instantiation and sets no
search path unless the user gave one.)

## Considered options

- **Runtime chain vs compile-time bake (chosen: chain).** The branch initially
  froze the build machine's `MaterialX_DIR` into the `.so` (two bakes:
  `MATERIALX_LIBRARIES_DIR` and `VISRTX_MATERIALX_MDL_DIR`). A baked path is a
  lie on any other machine and unfixable without a rebuild. Runtime resolution
  with an explicit-parameter override keeps installs relocatable and puts the
  choice with the user.
- **Device-global vs per-material search paths (chosen: device-global).** One
  stdlib serves every material; a per-material stdlib makes transcode cache
  identity messy and has no use case. Device parameter also mirrors the
  established `mdlSearchPaths` idiom.
- **Shipped builtin instantiation vs application-authored inline (chosen:
  inline).** The builtin alias made the device ship, install, and
  path-resolve a 7-line data file to save applications one generated string.
  Deleting it shrinks the device and removes a third baked path. Rejected:
  pointing applications at the distribution's nodedef file plus the
  auto-instantiate gate — fragile, layout-dependent, and the nodedef file
  carries no material node.
- **Application-side discovery (rejected).** TSD probing `/usr/share/MaterialX`
  et al. duplicates discovery already available through the MaterialX library
  the device links, drifts independently, and leaves every non-TSD ANARI
  application to reinvent it.

## Consequences

- Relocatable installs: no build-machine path survives into the binary's
  behavior on a correctly installed system.
- A misconfigured system (no parameter, no env, no discoverable install, stale
  bake) fails the transcode at commit with an error naming the chain — not a
  silent wrong-path fallback. Falling past an explicitly-set
  `materialxSearchPaths` to a later chain step warns, naming the winning step:
  the application asked for specific roots and is getting different ones.
- A root qualifies only when it holds `libraries/mdl` — the device consumes
  both the nodedefs and their MDL implementation modules, and an mdl-less root
  winning the chain would shadow later valid roots and fail much later as a
  cryptic `::materialx::*` import error.
- A root change re-transcodes committed `materialx` materials without the
  application touching them: the device pushes each live one back through the
  commit buffer (helium filters no-op commits, so a passive material would
  otherwise never observe the new root). The retranscode preserves bound
  textured inputs and the user's material selection — routing consumed those
  params, so the persisted copies are the only record of them. This is
  deliberately stronger than the `mdlSearchPaths` idiom (which leaves
  committed MDL materials compiled against the old paths). Known limit: the
  retranscode covers the `.mtlx` side only — when the switch is between
  *content-different* distributions whose generated MDL is byte-identical, the
  compiled-material cache (content-hashed inline module names) and the neuray
  module DB keep serving the old root's `::materialx::*` implementation
  modules. Relocation of a same-content distribution and unresolved→resolved
  recovery — this ADR's goals — are unaffected. Module eviction on a
  generation bump is follow-up work.
- The `VISRTX_MATERIAL_MATERIALX` parameter surface is unchanged
  (`source`/`sourceType`/`materialName`); the builtin-name special case
  disappears from `source` resolution.
- Tests that used `visrtx::standard_surface` author the inline instantiation
  instead.
- UI cosmetic: an application material editor shows raw XML in `source` for
  preset materials (inline document). Accepted.
