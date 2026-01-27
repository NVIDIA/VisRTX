## Lua Scripting

Enable with `cmake -DTSD_USE_LUA=ON ..` (auto-fetches Lua 5.4 + Sol3).

### Applications

- **tsdLua** — standalone interpreter (`./tsdLua script.lua`, `./tsdLua -e "..."`, `./tsdLua -i`)
- **tsdViewer** — embedded Lua terminal with live scene binding (requires `TSD_BUILD_INTERACTIVE_APPS=ON`)

Both provide a pre-bound `scene` variable.

### Actions Menu (tsdViewer)

`tsdViewer` populates its menu from Lua modules that call
`tsd.viewer.addMenuAction(path, fn)` during initialization. Each search path's
`init.lua` runs automatically, registering menu actions programmatically.

Lua module search paths (lowest to highest priority):

1. `<source>/tsd/scripts/` (dev builds only)
2. `<install>/share/tsd/scripts/`
3. `~/.config/tsd/scripts/` (Linux/macOS) or `%APPDATA%/tsd/scripts/` (Windows)
4. `TSD_LUA_PACKAGE_PATHS` env var (`:` separated on Unix, `;` on Windows)

Each path is added to Lua's `package.path` and its `init.lua` (if present) is
executed. Actions registered via `tsd.viewer.addMenuAction()` appear in the menu
tree, organized by `/`-separated path components.

```lua
-- Example: register a custom action in init.lua
tsd.viewer.addMenuAction("My Tools/Generate Spheres", function()
  tsd.io.generateRandomSpheres(scene)
end)
```

### API Quick Reference

All scripts have access to `scene` (a `tsd.Scene`) and the `tsd` module.
For the full API see [tsd.lua](tsd.lua) (LuaLS-annotated stub file).

```lua
-- Object creation
local geom = scene:createGeometry("triangle")
local mat  = scene:createMaterial("physicallyBased")
local surf = scene:createSurface("my_surface", geom, mat)

-- Parameters & arrays
mat:setParameter("baseColor", tsd.float3(0.8, 0.1, 0.1))
geom:setParameterArray("vertex.position", "float3", {
  {0, 0, 0}, {1, 0, 0}, {0, 1, 0}
})
geom:setParameterArray("primitive.index", "uint3", {{0, 1, 2}})

-- Math: float2/3/4, mat3 (packed SRT), mat4
local xfm = tsd.translation(tsd.float3(1, 0, 0)) * tsd.rotation(tsd.float3(0, 1, 0), tsd.radians(45))
local srt = tsd.srt(tsd.float3(1, 1, 1), tsd.float3(0, 45, 0), tsd.float3(1, 0, 0))

-- Layers & scene graph
local layer = scene:defaultLayer()
local node  = scene:insertChildTransformNode(layer:root(), xfm, "placed")
scene:insertObjectNode(node, surf)
node:setAsTransform(srt)                -- mat3 or mat4
local roundtrip = node:getTransformSRT() -- → mat3
scene:setOnlyLayerActive("default")

-- Import / export
tsd.io.importGLTF(scene, "model.gltf")
tsd.io.importHDRI(scene, "env.exr")
tsd.io.saveScene(scene, "out.tsd")
tsd.io.saveScene(scene, "out.tsd", { windows = { Viewport = { anariLibrary = "visrtx" } } })

-- Procedural generators
tsd.io.generateRandomSpheres(scene)
tsd.io.generateMaterialOrb(scene)

-- Batch rendering
local device = tsd.render.loadDevice("visrtx")
local ri = tsd.render.createRenderIndex(scene, device)
ri:populate()
local cam = tsd.CameraSetup.new()
cam.position, cam.direction, cam.up = tsd.float3(0, 0, 5), tsd.float3(0, 0, -1), tsd.float3(0, 1, 0)
cam.fovy, cam.aspect = 45.0, 16/9
local pl = tsd.render.createPipeline(1920, 1080, device, ri, cam)
tsd.render.renderToFile(pl, 128, "output.png", 1920, 1080)
```

### Example Scripts

See [scripts/examples/](../../../scripts/examples/) for worked examples:
`render_scene` (create HDRI dome, generate RTOW scene, render to file) and
`save_scene` (build an animated scene, save to `.tsd` with additional state).
