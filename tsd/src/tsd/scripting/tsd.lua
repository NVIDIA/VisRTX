---@meta

-- LuaLS stub file for the TSD scripting API
-- Generated from Sol2 bindings in tsd/src/tsd/scripting/bindings/

------------------------------------------------------------------------
-- Math types (MathBindings.cpp)
------------------------------------------------------------------------

---@class tsd.float2
---@field x number
---@field y number
---@operator add(tsd.float2): tsd.float2
---@operator sub(tsd.float2): tsd.float2
---@operator mul(number): tsd.float2
---@operator div(number): tsd.float2
local float2 = {}

---@class tsd.float3
---@field x number
---@field y number
---@field z number
---@operator add(tsd.float3): tsd.float3
---@operator sub(tsd.float3): tsd.float3
---@operator mul(number): tsd.float3
---@operator div(number): tsd.float3
local float3 = {}

---@class tsd.float4
---@field x number
---@field y number
---@field z number
---@field w number
---@operator add(tsd.float4): tsd.float4
---@operator sub(tsd.float4): tsd.float4
---@operator mul(number): tsd.float4
---@operator div(number): tsd.float4
local float4 = {}

---@class tsd.mat3
---@field identity tsd.mat3
---@operator index(integer): tsd.float3
local mat3 = {}

---@class tsd.mat4
---@field identity tsd.mat4
---@operator index(integer): tsd.float4
---@operator mul(tsd.mat4): tsd.mat4
---@operator mul(tsd.float4): tsd.float4
local mat4 = {}

------------------------------------------------------------------------
-- Parameter value types accepted by setParameter / returned by getParameter
------------------------------------------------------------------------

---@alias tsd.ParameterValue boolean|number|string|tsd.float2|tsd.float3|tsd.float4|tsd.mat4|tsd.Array|tsd.Sampler|tsd.Geometry|tsd.Material|tsd.SpatialField|tsd.Volume|tsd.Light|tsd.Camera|tsd.Surface|number[]

------------------------------------------------------------------------
-- Token (CoreBindings.cpp)
------------------------------------------------------------------------

---@class tsd.Token
local Token = {}

---@overload fun(): tsd.Token
---@overload fun(str: string): tsd.Token
---@return tsd.Token
function Token.new(...) end

---@return string
function Token:str() end

---@return boolean
function Token:empty() end

------------------------------------------------------------------------
-- Parameter (CoreBindings.cpp — read-only descriptor)
------------------------------------------------------------------------

---@class tsd.Parameter
local Parameter = {}

---@return string
function Parameter:name() end

---@return string
function Parameter:description() end

---@return boolean
function Parameter:isEnabled() end

------------------------------------------------------------------------
-- Object (ObjectMethodBindings.hpp — base type for all scene objects)
------------------------------------------------------------------------

---@class tsd.Object
---@field name string # Object name (read/write)
local Object = {}

---@return string
function Object:subtype() end

---@return integer
function Object:type() end

---@return integer
function Object:index() end

--- Set a parameter value on this object.
---@param name string
---@param value tsd.ParameterValue
function Object:setParameter(name, value) end

--- Create an array, populate it from Lua data, and bind it to a parameter.
--- For 2D/3D arrays, `data` may be linear or shape-matching nested.
---@param name string
---@param typeStr string
---@param data table
---@overload fun(self: tsd.Object, name: string, typeStr: string, items0: integer, data: table): tsd.Array
---@overload fun(self: tsd.Object, name: string, typeStr: string, items0: integer, items1: integer, data: table): tsd.Array
---@overload fun(self: tsd.Object, name: string, typeStr: string, items0: integer, items1: integer, items2: integer, data: table): tsd.Array
---@return tsd.Array
function Object:setParameterArray(name, typeStr, data) end

--- Get the Parameter descriptor for the named parameter.
---@param name string
---@return tsd.Parameter?
function Object:parameter(name) end

--- Get the current value of a parameter.
---@param name string
---@return tsd.ParameterValue?
function Object:getParameter(name) end

---@param name string
function Object:removeParameter(name) end

function Object:removeAllParameters() end

---@return integer
function Object:numParameters() end

---@param index integer
---@return string
function Object:parameterNameAt(index) end

--- Set a metadata value on this object.
---@param key string
---@param value boolean|number|string
function Object:setMetadata(key, value) end

--- Get a metadata value from this object.
---@param key string
---@return boolean|number|string|nil
function Object:getMetadata(key) end

---@param key string
function Object:removeMetadata(key) end

---@return integer
function Object:numMetadata() end

---@param index integer
---@return string
function Object:getMetadataName(index) end

------------------------------------------------------------------------
-- Object types (ObjectBindings.cpp)
------------------------------------------------------------------------

---@class tsd.Geometry: tsd.Object
local Geometry = {}
---@return boolean
function Geometry:valid() end

---@class tsd.Material: tsd.Object
local Material = {}
---@return boolean
function Material:valid() end

---@class tsd.Light: tsd.Object
local Light = {}
---@return boolean
function Light:valid() end

---@class tsd.Camera: tsd.Object
local Camera = {}
---@return boolean
function Camera:valid() end

---@class tsd.Sampler: tsd.Object
local Sampler = {}
---@return boolean
function Sampler:valid() end

---@class tsd.Surface: tsd.Object
local Surface = {}
---@return boolean
function Surface:valid() end
--- Get the geometry attached to this surface.
---@return tsd.Geometry?
function Surface:geometry() end
--- Get the material attached to this surface.
---@return tsd.Material?
function Surface:material() end

---@class tsd.Volume: tsd.Object
local Volume = {}
---@return boolean
function Volume:valid() end
--- Get the spatial field attached to this volume.
---@return tsd.SpatialField?
function Volume:spatialField() end

---@class tsd.SpatialField: tsd.Object
local SpatialField = {}
---@return boolean
function SpatialField:valid() end
--- Compute the value range of this spatial field.
---@return tsd.float2
function SpatialField:computeValueRange() end

---@class tsd.Array: tsd.Object
local Array = {}
---@return boolean
function Array:valid() end
---@return integer
function Array:elementType() end
---@return integer
function Array:size() end
---@return integer
function Array:elementSize() end
---@return boolean
function Array:isEmpty() end
---@param d integer
---@return integer
function Array:dim(d) end
--- Set array data from a Lua table.
--- For 2D/3D arrays, supports either a flat/linear table or a shape-matching nested table.
--- Vector elements support `tsd.float2/3/4(...)` values or numeric tables.
---@param data table
function Array:setData(data) end
--- Get array data as a flat/linear Lua table.
---@return table
function Array:getData() end

------------------------------------------------------------------------
-- Layer types (LayerBindings.cpp)
------------------------------------------------------------------------

---@class tsd.LayerNode
---@field name string # Node name (read/write)
local LayerNode = {}

---@return boolean
function LayerNode:valid() end

---@return integer
function LayerNode:index() end

---@return tsd.LayerNode
function LayerNode:parent() end

---@return tsd.LayerNode
function LayerNode:next() end

---@return tsd.LayerNode
function LayerNode:sibling() end

---@return boolean
function LayerNode:isRoot() end

---@return boolean
function LayerNode:isLeaf() end

--- Get the Nth direct child (0-based). Returns invalid node if out of range.
---@param index integer
---@return tsd.LayerNode
function LayerNode:child(index) end

--- Find the first direct child with the given name.
---@param name string
---@return tsd.LayerNode
function LayerNode:childByName(name) end

---@return boolean
function LayerNode:isObject() end

---@return boolean
function LayerNode:isTransform() end

---@return boolean
function LayerNode:isEmpty() end

---@return boolean
function LayerNode:isEnabled() end

---@param enabled boolean
function LayerNode:setEnabled(enabled) end

---@return tsd.mat4
function LayerNode:getTransform() end

--- Get transform as packed SRT (columns: scale, euler-rotation-degrees, translation).
---@return tsd.mat3
function LayerNode:getTransformSRT() end

---@overload fun(self: tsd.LayerNode, m: tsd.mat4)
---@overload fun(self: tsd.LayerNode, srt: tsd.mat3)
function LayerNode:setAsTransform(m) end

--- Set node as a transform array (array of mat4 for multi-instancing).
---@param a tsd.Array
function LayerNode:setAsTransformArray(a) end

--- Get the transform array, or nil if this node is not a transform array.
---@return tsd.Array|nil
function LayerNode:getTransformArray() end

---@class tsd.Layer
local Layer = {}

---@return tsd.LayerNode
function Layer:root() end

---@return integer
function Layer:size() end

---@return boolean
function Layer:empty() end

---@param index integer
---@return tsd.LayerNode
function Layer:at(index) end

--- Traverse the layer tree. Callback receives (node, level). Return false to stop.
---@param fn fun(node: tsd.LayerNode, level: integer): boolean?
function Layer:foreach(fn) end

------------------------------------------------------------------------
-- Animation (CoreBindings.cpp)
------------------------------------------------------------------------

---@class tsd.Animation
---@field name string # Animation name (read/write)
local Animation = {}

---@return string
function Animation:info() end

---@return integer
function Animation:timeStepCount() end

---@param time number
function Animation:update(time) end

--- Bind time-step arrays to an object's parameters for animation.
--- Single parameter: pass a string name and a single Array.
--- Multi parameter: pass a table of string names and a table of Arrays.
---@overload fun(self: tsd.Animation, obj: tsd.Object, param: string, array: tsd.Array)
---@overload fun(self: tsd.Animation, obj: tsd.Object, params: string[], arrays: tsd.Array[])
function Animation:setAsTimeSteps(obj, params, arrays) end

------------------------------------------------------------------------
-- Scene (CoreBindings.cpp)
------------------------------------------------------------------------

---@class tsd.Scene
local Scene = {}

---@return tsd.Scene
function Scene.new() end

-- Object creation --------------------------------------------------------

---@param subtype string
---@param params? table<string, tsd.ParameterValue>
---@return tsd.Geometry
function Scene:createGeometry(subtype, params) end

---@param subtype string
---@param params? table<string, tsd.ParameterValue>
---@return tsd.Material
function Scene:createMaterial(subtype, params) end

---@param subtype string
---@param params? table<string, tsd.ParameterValue>
---@return tsd.Light
function Scene:createLight(subtype, params) end

---@param subtype string
---@param params? table<string, tsd.ParameterValue>
---@return tsd.Camera
function Scene:createCamera(subtype, params) end

---@param subtype string
---@param params? table<string, tsd.ParameterValue>
---@return tsd.Sampler
function Scene:createSampler(subtype, params) end

---@param subtype string
---@param params? table<string, tsd.ParameterValue>
---@return tsd.Volume
function Scene:createVolume(subtype, params) end

---@param subtype string
---@param params? table<string, tsd.ParameterValue>
---@return tsd.SpatialField
function Scene:createSpatialField(subtype, params) end

---@param name string
---@param geometry tsd.Geometry
---@param material tsd.Material
---@param params? table<string, tsd.ParameterValue>
---@return tsd.Surface
function Scene:createSurface(name, geometry, material, params) end

--- Create a typed array. Valid type strings:
--- "float", "float2", "float3", "float4",
--- "int", "int2", "int3", "int4",
--- "uint", "uint2", "uint3", "uint4",
--- "mat4".
---@param typeStr string
---@param items0 integer
---@overload fun(self: tsd.Scene, typeStr: string, items0: integer, items1: integer): tsd.Array
---@overload fun(self: tsd.Scene, typeStr: string, items0: integer, items1: integer, items2: integer): tsd.Array
---@return tsd.Array
function Scene:createArray(typeStr, items0) end

-- Object access ----------------------------------------------------------

---@param index integer
---@return tsd.Geometry
function Scene:getGeometry(index) end

---@param index integer
---@return tsd.Material
function Scene:getMaterial(index) end

---@param index integer
---@return tsd.Light
function Scene:getLight(index) end

---@param index integer
---@return tsd.Camera
function Scene:getCamera(index) end

---@param index integer
---@return tsd.Surface
function Scene:getSurface(index) end

---@param index integer
---@return tsd.Array
function Scene:getArray(index) end

---@param index integer
---@return tsd.Volume
function Scene:getVolume(index) end

---@param index integer
---@return tsd.Sampler
function Scene:getSampler(index) end

---@param index integer
---@return tsd.SpatialField
function Scene:getSpatialField(index) end

-- Object counts ----------------------------------------------------------

--- Return the number of objects of the given ANARI type.
---@param type integer # Use tsd.GEOMETRY, tsd.MATERIAL, etc.
---@return integer
function Scene:numberOfObjects(type) end

-- Iteration --------------------------------------------------------------

--- Iterate over all geometries. Return false from the callback to stop early.
---@param fn fun(obj: tsd.Geometry): boolean?
function Scene:forEachGeometry(fn) end

--- Iterate over all materials. Return false from the callback to stop early.
---@param fn fun(obj: tsd.Material): boolean?
function Scene:forEachMaterial(fn) end

--- Iterate over all surfaces. Return false from the callback to stop early.
---@param fn fun(obj: tsd.Surface): boolean?
function Scene:forEachSurface(fn) end

--- Iterate over all lights. Return false from the callback to stop early.
---@param fn fun(obj: tsd.Light): boolean?
function Scene:forEachLight(fn) end

--- Iterate over all cameras. Return false from the callback to stop early.
---@param fn fun(obj: tsd.Camera): boolean?
function Scene:forEachCamera(fn) end

--- Iterate over all volumes. Return false from the callback to stop early.
---@param fn fun(obj: tsd.Volume): boolean?
function Scene:forEachVolume(fn) end

--- Iterate over all spatial fields. Return false from the callback to stop early.
---@param fn fun(obj: tsd.SpatialField): boolean?
function Scene:forEachSpatialField(fn) end

--- Iterate over all samplers. Return false from the callback to stop early.
---@param fn fun(obj: tsd.Sampler): boolean?
function Scene:forEachSampler(fn) end

--- Iterate over all arrays. Return false from the callback to stop early.
---@param fn fun(obj: tsd.Array): boolean?
function Scene:forEachArray(fn) end

-- Layers -----------------------------------------------------------------

---@param name string
---@return tsd.Layer
function Scene:addLayer(name) end

---@overload fun(self: tsd.Scene, name: string): tsd.Layer
---@overload fun(self: tsd.Scene, index: integer): tsd.Layer
---@return tsd.Layer
function Scene:layer(...) end

---@return integer
function Scene:numberOfLayers() end

---@return tsd.Layer
function Scene:defaultLayer() end

---@return tsd.Material
function Scene:defaultMaterial() end

---@param name string
---@return boolean
function Scene:layerIsActive(name) end

---@param name string
---@param active boolean
function Scene:setLayerActive(name, active) end

function Scene:setAllLayersActive() end

--- Deactivate all layers, then activate only the named layer.
---@param name string
function Scene:setOnlyLayerActive(name) end

---@return integer
function Scene:numberOfActiveLayers() end

--- Signal that a layer has changed (needed after modifying transforms).
---@param layer tsd.Layer
function Scene:signalLayerChange(layer) end

-- Node insertion ---------------------------------------------------------

---@param parent tsd.LayerNode
---@param name string
---@return tsd.LayerNode
function Scene:insertChildNode(parent, name) end

---@param parent tsd.LayerNode
---@param transform tsd.mat4
---@param name string
---@return tsd.LayerNode
function Scene:insertChildTransformNode(parent, transform, name) end

--- Insert a child node with an array of mat4 transforms (multi-instancing).
---@param parent tsd.LayerNode
---@param array tsd.Array # Array of FLOAT32_MAT4
---@param name string
---@return tsd.LayerNode
function Scene:insertChildTransformArrayNode(parent, array, name) end

--- Insert an object (surface, light, or volume) into the scene graph.
---@param parent tsd.LayerNode
---@param obj tsd.Surface|tsd.Light|tsd.Volume
---@param name? string
---@return tsd.LayerNode
function Scene:insertObjectNode(parent, obj, name) end

-- Object removal ---------------------------------------------------------

---@param obj tsd.Object
function Scene:removeObject(obj) end

function Scene:removeAllObjects() end

---@overload fun(self: tsd.Scene, name: string)
---@overload fun(self: tsd.Scene, layer: tsd.Layer)
function Scene:removeLayer(...) end

function Scene:removeAllLayers() end

--- Remove a node from the scene graph.
---@overload fun(self: tsd.Scene, node: tsd.LayerNode)
---@overload fun(self: tsd.Scene, node: tsd.LayerNode, deleteObjects: boolean)
function Scene:removeNode(...) end

-- Animation --------------------------------------------------------------

---@overload fun(self: tsd.Scene): tsd.Animation
---@overload fun(self: tsd.Scene, name: string): tsd.Animation
---@return tsd.Animation
function Scene:addAnimation(...) end

---@return integer
function Scene:numberOfAnimations() end

---@param index integer
---@return tsd.Animation
function Scene:animation(index) end

---@param animation tsd.Animation
function Scene:removeAnimation(animation) end

function Scene:removeAllAnimations() end

---@param time number
function Scene:setAnimationTime(time) end

---@return number
function Scene:getAnimationTime() end

---@param increment number
function Scene:setAnimationIncrement(increment) end

---@return number
function Scene:getAnimationIncrement() end

function Scene:incrementAnimationTime() end

-- Cleanup ----------------------------------------------------------------

function Scene:removeUnusedObjects() end

function Scene:defragmentObjectStorage() end

function Scene:cleanupScene() end

------------------------------------------------------------------------
-- Render types (RenderBindings.cpp)
------------------------------------------------------------------------

---@class tsd.CameraSetup
---@field position tsd.float3
---@field direction tsd.float3
---@field up tsd.float3
---@field fovy number
---@field aspect number
local CameraSetup = {}

---@return tsd.CameraSetup
function CameraSetup.new() end

---@class tsd.AnariDevice
---@field libraryName string # (read-only)

---@class tsd.RenderIndex
local RenderIndex = {}

---@overload fun(scene: tsd.Scene, device: any): tsd.RenderIndex
---@return tsd.RenderIndex
function RenderIndex.new(...) end

function RenderIndex:populate() end

---@return any
function RenderIndex:world() end

---@return any
function RenderIndex:device() end

---@class tsd.RenderPipeline
local RenderPipeline = {}

---@overload fun(): tsd.RenderPipeline
---@overload fun(width: integer, height: integer): tsd.RenderPipeline
---@return tsd.RenderPipeline
function RenderPipeline.new(...) end

---@param width integer
---@param height integer
function RenderPipeline:setDimensions(width, height) end

function RenderPipeline:render() end

---@return integer
function RenderPipeline:size() end

---@return boolean
function RenderPipeline:empty() end

function RenderPipeline:clear() end

------------------------------------------------------------------------
-- Module-level table (injected as a global by the C++ runtime)
------------------------------------------------------------------------

---@class tsd
---@diagnostic disable-next-line: lowercase-global
tsd = {}

-- Scene creation ---------------------------------------------------------

---@return tsd.Scene
function tsd.createScene() end

-- ANARI data type constants ----------------------------------------------

---@type integer
tsd.GEOMETRY = 0
---@type integer
tsd.MATERIAL = 0
---@type integer
tsd.LIGHT = 0
---@type integer
tsd.CAMERA = 0
---@type integer
tsd.SURFACE = 0
---@type integer
tsd.VOLUME = 0
---@type integer
tsd.SAMPLER = 0
---@type integer
tsd.ARRAY = 0
---@type integer
tsd.SPATIAL_FIELD = 0

-- Math utility functions (MathBindings.cpp) ------------------------------

--- Construct a float2.
---@overload fun(): tsd.float2
---@overload fun(x: number, y: number): tsd.float2
---@return tsd.float2
function tsd.float2(...) end

--- Construct a float3.
---@overload fun(): tsd.float3
---@overload fun(x: number, y: number, z: number): tsd.float3
---@return tsd.float3
function tsd.float3(...) end

--- Construct a float4.
---@overload fun(): tsd.float4
---@overload fun(x: number, y: number, z: number, w: number): tsd.float4
---@return tsd.float4
function tsd.float4(...) end

--- Construct a mat3 (packed SRT: columns = scale, euler-rotation-degrees, translation).
---@overload fun(): tsd.mat3
---@overload fun(col0: tsd.float3, col1: tsd.float3, col2: tsd.float3): tsd.mat3
---@return tsd.mat3
function tsd.mat3(...) end

--- Construct a mat4.
---@overload fun(): tsd.mat4
---@overload fun(col0: tsd.float4, col1: tsd.float4, col2: tsd.float4, col3: tsd.float4): tsd.mat4
---@return tsd.mat4
function tsd.mat4(...) end

--- Alias for tsd.mat3 — construct a packed SRT matrix.
---@overload fun(): tsd.mat3
---@param scale tsd.float3
---@param rotation tsd.float3 # Euler rotation in degrees
---@param translation tsd.float3
---@return tsd.mat3
function tsd.srt(scale, rotation, translation) end

--- Compute the length of a vector.
---@overload fun(v: tsd.float2): number
---@overload fun(v: tsd.float3): number
---@overload fun(v: tsd.float4): number
---@return number
function tsd.length(v) end

--- Normalize a vector to unit length.
---@overload fun(v: tsd.float2): tsd.float2
---@overload fun(v: tsd.float3): tsd.float3
---@overload fun(v: tsd.float4): tsd.float4
function tsd.normalize(v) end

--- Compute the dot product of two vectors.
---@overload fun(a: tsd.float2, b: tsd.float2): number
---@overload fun(a: tsd.float3, b: tsd.float3): number
---@overload fun(a: tsd.float4, b: tsd.float4): number
---@return number
function tsd.dot(a, b) end

--- Compute the cross product of two float3 vectors.
---@param a tsd.float3
---@param b tsd.float3
---@return tsd.float3
function tsd.cross(a, b) end

--- Create a translation matrix.
---@param t tsd.float3
---@return tsd.mat4
function tsd.translation(t) end

--- Create a scaling matrix.
---@overload fun(s: tsd.float3): tsd.mat4
---@overload fun(s: number): tsd.mat4
---@return tsd.mat4
function tsd.scaling(s) end

--- Create a rotation matrix from an axis and angle (in radians).
---@param axis tsd.float3
---@param angle number
---@return tsd.mat4
function tsd.rotation(axis, angle) end

--- Convert degrees to radians.
---@param degrees number
---@return number
function tsd.radians(degrees) end

--- Convert radians to degrees.
---@param radians number
---@return number
function tsd.degrees(radians) end
------------------------------------------------------------------------
-- Sub-tables
------------------------------------------------------------------------

-- tsd.viewer --------------------------------------------------------------
-- NOTE: tsd.viewer is only available inside the interactive viewer (tsdViewer).
-- In standalone tsdLua, tsd.viewer is nil.  Guard with: if tsd.viewer then ... end

---@class tsd.viewer
tsd.viewer = {}

--- Request a viewer refresh (re-render the current frame).
function tsd.viewer.refresh() end

--- Register a menu action in the Lua menu.
--- The path uses `/` separators to define the menu hierarchy (e.g. "Import/glTF/Box").
---@param path string Menu path (e.g. "Category/Subcategory/Action Name")
---@param fn function The function to call when the action is selected
function tsd.viewer.addMenuAction(path, fn) end

--- Add a separator in the Lua menu under the given category path.
---@param categoryPath string Menu path for the separator (e.g. "Category/Subcategory")
function tsd.viewer.addSeparator(categoryPath) end

--- Clear all registered actions from the Lua menu.
function tsd.viewer.clearActions() end

-- tsd.io (IOBindings.cpp) ------------------------------------------------

---@class tsd.io
tsd.io = {}

--- Import an OBJ file.
---@overload fun(scene: tsd.Scene, filename: string)
---@overload fun(scene: tsd.Scene, filename: string, location: tsd.LayerNode)
---@overload fun(scene: tsd.Scene, filename: string, location: tsd.LayerNode, useDefaultMat: boolean)
function tsd.io.importOBJ(...) end

--- Import a glTF file.
---@overload fun(scene: tsd.Scene, filename: string)
---@overload fun(scene: tsd.Scene, filename: string, location: tsd.LayerNode)
function tsd.io.importGLTF(...) end

--- Import a PLY file.
---@overload fun(scene: tsd.Scene, filename: string)
---@overload fun(scene: tsd.Scene, filename: string, location: tsd.LayerNode)
function tsd.io.importPLY(...) end

--- Import an HDRI environment map.
---@overload fun(scene: tsd.Scene, filename: string)
---@overload fun(scene: tsd.Scene, filename: string, location: tsd.LayerNode)
function tsd.io.importHDRI(...) end

--- Import a USD file.
---@overload fun(scene: tsd.Scene, filename: string)
---@overload fun(scene: tsd.Scene, filename: string, location: tsd.LayerNode)
function tsd.io.importUSD(...) end

--- Import a PDB (Protein Data Bank) file.
---@overload fun(scene: tsd.Scene, filename: string)
---@overload fun(scene: tsd.Scene, filename: string, location: tsd.LayerNode)
function tsd.io.importPDB(...) end

--- Import an SWC (neuron morphology) file.
---@overload fun(scene: tsd.Scene, filename: string)
---@overload fun(scene: tsd.Scene, filename: string, location: tsd.LayerNode)
function tsd.io.importSWC(...) end

--- Import a volume file (auto-detects format).
---@overload fun(scene: tsd.Scene, filename: string): tsd.Volume
---@overload fun(scene: tsd.Scene, filename: string, location: tsd.LayerNode): tsd.Volume
function tsd.io.importVolume(...) end

--- Import a RAW volume file.
---@param scene tsd.Scene
---@param filename string
---@return tsd.Volume
function tsd.io.importRAW(scene, filename) end

--- Import a NanoVDB volume file.
---@param scene tsd.Scene
---@param filename string
---@return tsd.Volume
function tsd.io.importNVDB(scene, filename) end

--- Import an MHD (MetaImage) volume file.
---@param scene tsd.Scene
---@param filename string
---@return tsd.Volume
function tsd.io.importMHD(scene, filename) end

--- Generate random spheres.
---@overload fun(scene: tsd.Scene)
---@overload fun(scene: tsd.Scene, location: tsd.LayerNode)
---@overload fun(scene: tsd.Scene, location: tsd.LayerNode, useDefaultMat: boolean)
function tsd.io.generateRandomSpheres(...) end

--- Generate a material test orb.
---@overload fun(scene: tsd.Scene)
---@overload fun(scene: tsd.Scene, location: tsd.LayerNode)
function tsd.io.generateMaterialOrb(...) end

--- Generate a Blender monkey (Suzanne).
---@overload fun(scene: tsd.Scene)
---@overload fun(scene: tsd.Scene, location: tsd.LayerNode)
function tsd.io.generateMonkey(...) end

--- Generate sample cylinders.
---@overload fun(scene: tsd.Scene)
---@overload fun(scene: tsd.Scene, location: tsd.LayerNode)
---@overload fun(scene: tsd.Scene, location: tsd.LayerNode, useDefaultMat: boolean)
function tsd.io.generateCylinders(...) end

--- Generate default scene lights.
---@param scene tsd.Scene
function tsd.io.generateDefaultLights(scene) end

--- Generate an HDRI dome light.
---@overload fun(scene: tsd.Scene)
---@overload fun(scene: tsd.Scene, location: tsd.LayerNode)
function tsd.io.generateHdriDome(...) end

--- Generate the "Ray Tracing in One Weekend" scene.
---@overload fun(scene: tsd.Scene)
---@overload fun(scene: tsd.Scene, location: tsd.LayerNode)
function tsd.io.generateRtow(...) end

--- Generate a sphere-set volume.
---@overload fun(scene: tsd.Scene)
---@overload fun(scene: tsd.Scene, location: tsd.LayerNode)
function tsd.io.generateSphereSetVolume(...) end

--- Save a scene to a TSD file.
--- When called with a state table, the file can be opened directly in
--- tsdViewer with the correct device and camera position.
---@overload fun(scene: tsd.Scene, filename: string)
---@overload fun(scene: tsd.Scene, filename: string, state: table)
function tsd.io.saveScene(...) end

--- Load a scene from a TSD file.
---@param scene tsd.Scene
---@param filename string
function tsd.io.loadScene(scene, filename) end

-- tsd.render (RenderBindings.cpp) ----------------------------------------

---@class tsd.render
tsd.render = {}

--- Load an ANARI device by library name.
---@param libraryName string
---@return tsd.AnariDevice
function tsd.render.loadDevice(libraryName) end

--- Create a render index for a scene.
--- Throws if `device` is nil or invalid.
---@param scene tsd.Scene
---@param device tsd.AnariDevice
---@return tsd.RenderIndex
function tsd.render.createRenderIndex(scene, device) end

--- Get the world bounds from a render index.
--- Throws if `device` or `index` is nil or invalid.
---@param device tsd.AnariDevice
---@param index tsd.RenderIndex
---@return {min: tsd.float3, max: tsd.float3}
function tsd.render.getWorldBounds(device, index) end

--- Create a render pipeline with a scene render pass.
--- Throws if width/height are <= 0, or if `device`/`index` are nil/invalid.
---@param width integer
---@param height integer
---@param device tsd.AnariDevice
---@param index tsd.RenderIndex
---@param camera tsd.CameraSetup
---@param rendererParams? table<string, boolean|number|string>  Optional renderer parameters (e.g. {denoise=true, denoiseMode="colorAlbedoNormal"})
---@return tsd.RenderPipeline
function tsd.render.createPipeline(width, height, device, index, camera, rendererParams) end

--- Render multiple samples and save to an image file.
--- Supported formats: png, jpg/jpeg, bmp, tga, ppm.
--- Throws if `pipeline` is nil, `samples < 1`, or width/height are <= 0.
--- The pipeline dimensions are set to `(width, height)` before rendering.
---@param pipeline tsd.RenderPipeline
---@param samples integer
---@param filename string
---@param width integer
---@param height integer
function tsd.render.renderToFile(pipeline, samples, filename, width, height) end

------------------------------------------------------------------------
-- Global variable: the pre-bound scene instance
------------------------------------------------------------------------

---@type tsd.Scene
scene = nil

return tsd
