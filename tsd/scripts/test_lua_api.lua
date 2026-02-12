-- Comprehensive TSD Lua API Test Script
-- Tests all documented API functions to verify they work correctly

local passed = 0
local failed = 0
local errors = {}

local function test(name, fn)
  local ok, err = pcall(fn)
  if ok then
    passed = passed + 1
    print("[PASS] " .. name)
  else
    failed = failed + 1
    table.insert(errors, {name = name, error = tostring(err)})
    print("[FAIL] " .. name .. ": " .. tostring(err))
  end
end

local function section(name)
  print("\n=== " .. name .. " ===")
end

-- Check that scene is bound
section("Prerequisites")
test("scene is bound", function()
  assert(scene ~= nil, "scene should be bound")
end)

test("tsd module exists", function()
  assert(tsd ~= nil, "tsd module should exist")
end)

-- Vector types
section("Vector Types")

test("tsd.float2(x, y)", function()
  local v = tsd.float2(1.0, 2.0)
  assert(v ~= nil, "float2 should be created")
  assert(v.x == 1.0, "v.x should be 1.0")
  assert(v.y == 2.0, "v.y should be 2.0")
end)

test("tsd.float2()", function()
  local v = tsd.float2()
  assert(v ~= nil, "float2 should be created")
  assert(v.x == 0.0, "v.x should default to 0.0")
  assert(v.y == 0.0, "v.y should default to 0.0")
end)

test("tsd.float3()", function()
  local v = tsd.float3()
  assert(v ~= nil, "float3 should be created")
  assert(v.x == 0.0, "v.x should default to 0.0")
  assert(v.y == 0.0, "v.y should default to 0.0")
  assert(v.z == 0.0, "v.z should default to 0.0")
end)

test("tsd.float3(x, y, z)", function()
  local v = tsd.float3(1.0, 2.0, 3.0)
  assert(v ~= nil, "float3 should be created")
  assert(v.x == 1.0, "v.x should be 1.0")
  assert(v.y == 2.0, "v.y should be 2.0")
  assert(v.z == 3.0, "v.z should be 3.0")
end)

test("tsd.float3(v) uniform is not supported", function()
  local ok = pcall(function()
    tsd.float3(5.0)
  end)
  assert(not ok, "float3(v) should not be supported")
end)

test("tsd.float4()", function()
  local v = tsd.float4()
  assert(v ~= nil, "float4 should be created")
  assert(v.x == 0.0, "v.x should default to 0.0")
  assert(v.y == 0.0, "v.y should default to 0.0")
  assert(v.z == 0.0, "v.z should default to 0.0")
  assert(v.w == 0.0, "v.w should default to 0.0")
end)

test("tsd.float4(x, y, z, w)", function()
  local v = tsd.float4(1.0, 2.0, 3.0, 4.0)
  assert(v ~= nil, "float4 should be created")
  assert(v.x == 1.0, "v.x should be 1.0")
  assert(v.w == 4.0, "v.w should be 4.0")
end)

test("tsd.float4(float3, w) is not supported", function()
  local ok = pcall(function()
    tsd.float4(tsd.float3(1.0, 2.0, 3.0), 4.0)
  end)
  assert(not ok, "float4(float3, w) should not be supported")
end)

test("tsd.float4(v) uniform is not supported", function()
  local ok = pcall(function()
    tsd.float4(5.0)
  end)
  assert(not ok, "float4(v) should not be supported")
end)

-- Vector operations
section("Vector Operations")

test("tsd.length(float3)", function()
  local v = tsd.float3(3.0, 4.0, 0.0)
  local len = tsd.length(v)
  assert(math.abs(len - 5.0) < 0.001, "length should be 5.0")
end)

test("tsd.normalize(float3)", function()
  local v = tsd.float3(3.0, 4.0, 0.0)
  local n = tsd.normalize(v)
  assert(n ~= nil, "normalized vector should exist")
  local len = tsd.length(n)
  assert(math.abs(len - 1.0) < 0.001, "normalized length should be 1.0")
end)

test("tsd.dot(float3, float3)", function()
  local a = tsd.float3(1.0, 0.0, 0.0)
  local b = tsd.float3(0.0, 1.0, 0.0)
  local d = tsd.dot(a, b)
  assert(math.abs(d) < 0.001, "dot product of perpendicular vectors should be 0")
end)

test("tsd.cross(float3, float3)", function()
  local a = tsd.float3(1.0, 0.0, 0.0)
  local b = tsd.float3(0.0, 1.0, 0.0)
  local c = tsd.cross(a, b)
  assert(c ~= nil, "cross product should exist")
  assert(math.abs(c.z - 1.0) < 0.001, "cross product z should be 1.0")
end)

test("tsd.radians(degrees)", function()
  local r = tsd.radians(180.0)
  assert(math.abs(r - math.pi) < 0.001, "180 degrees should be pi radians")
end)

test("tsd.degrees(radians)", function()
  local d = tsd.degrees(math.pi)
  assert(math.abs(d - 180.0) < 0.001, "pi radians should be 180 degrees")
end)

-- Transform matrices
section("Transform Matrices")

test("tsd.translation(float3)", function()
  local t = tsd.translation(tsd.float3(1.0, 2.0, 3.0))
  assert(t ~= nil, "translation matrix should be created")
end)

test("tsd.scaling(float3)", function()
  local s = tsd.scaling(tsd.float3(2.0, 2.0, 2.0))
  assert(s ~= nil, "scaling matrix should be created")
end)

test("tsd.scaling(float)", function()
  local s = tsd.scaling(2.0)
  assert(s ~= nil, "uniform scaling matrix should be created")
end)

test("tsd.rotation(axis, angle)", function()
  local r = tsd.rotation(tsd.float3(0.0, 1.0, 0.0), tsd.radians(90.0))
  assert(r ~= nil, "rotation matrix should be created")
end)

test("matrix multiplication", function()
  local t = tsd.translation(tsd.float3(1.0, 0.0, 0.0))
  local s = tsd.scaling(2.0)
  local combined = t * s
  assert(combined ~= nil, "matrix multiplication should work")
end)

test("matrix-vector multiplication", function()
  local v = tsd.float4(1.0, 2.0, 3.0, 1.0)
  local out = tsd.mat4.identity * v
  assert(math.abs(out.x - 1.0) < 0.001 and math.abs(out.y - 2.0) < 0.001
      and math.abs(out.z - 3.0) < 0.001 and math.abs(out.w - 1.0) < 0.001,
      "mat4 * float4 should work")
end)

-- Geometry creation
section("Geometry Creation")

test("scene:createGeometry('sphere')", function()
  local geom = scene:createGeometry("sphere")
  assert(geom ~= nil, "sphere geometry should be created")
  assert(geom:valid(), "geometry ref should be valid")
end)

test("scene:createGeometry('cylinder')", function()
  local geom = scene:createGeometry("cylinder")
  assert(geom ~= nil, "cylinder geometry should be created")
end)

test("scene:createGeometry('cone')", function()
  local geom = scene:createGeometry("cone")
  assert(geom ~= nil, "cone geometry should be created")
end)

test("scene:createGeometry('quad')", function()
  local geom = scene:createGeometry("quad")
  assert(geom ~= nil, "quad geometry should be created")
end)

test("scene:createGeometry('triangle')", function()
  local geom = scene:createGeometry("triangle")
  assert(geom ~= nil, "triangle geometry should be created")
end)

-- Material creation
section("Material Creation")

test("scene:createMaterial('matte')", function()
  local mat = scene:createMaterial("matte")
  assert(mat ~= nil, "matte material should be created")
  assert(mat:valid(), "material ref should be valid")
end)

test("scene:createMaterial('physicallyBased')", function()
  local mat = scene:createMaterial("physicallyBased")
  assert(mat ~= nil, "physicallyBased material should be created")
end)

-- Light creation
section("Light Creation")

test("scene:createLight('point')", function()
  local light = scene:createLight("point")
  assert(light ~= nil, "point light should be created")
end)

test("scene:createLight('directional')", function()
  local light = scene:createLight("directional")
  assert(light ~= nil, "directional light should be created")
end)

test("scene:createLight('spot')", function()
  local light = scene:createLight("spot")
  assert(light ~= nil, "spot light should be created")
end)

-- Parameter setting
section("Parameter Setting")

test("ref:setParameter with float", function()
  local geom = scene:createGeometry("sphere")
  geom:setParameter("radius", 1.5)
end)

test("ref:setParameter with float3", function()
  local mat = scene:createMaterial("matte")
  mat:setParameter("color", tsd.float3(1.0, 0.0, 0.0))
end)

test("ref:setParameter with table as float3", function()
  local mat = scene:createMaterial("matte")
  mat:setParameter("color", {0.5, 0.5, 0.5})
  local c = mat:getParameter("color")
  assert(c ~= nil, "color should be set")
  assert(math.abs(c.x - 0.5) < 0.001, "color.x should be 0.5")
end)

test("ref:setParameter with bool", function()
  local geom = scene:createGeometry("cylinder")
  geom:setParameter("caps", true)
  local val = geom:getParameter("caps")
  assert(val == true, "caps should be true")
end)

test("ref.name property", function()
  local geom = scene:createGeometry("sphere")
  geom.name = "myGeometry"
  assert(geom.name == "myGeometry", "name should be set")
end)

test("ref:subtype()", function()
  local geom = scene:createGeometry("sphere")
  assert(geom:subtype() == "sphere", "subtype should be 'sphere'")
end)

test("ref:type()", function()
  local geom = scene:createGeometry("sphere")
  assert(geom:type() == tsd.GEOMETRY, "type should be GEOMETRY")
  local mat = scene:createMaterial("matte")
  assert(mat:type() == tsd.MATERIAL, "type should be MATERIAL")
  local light = scene:createLight("point")
  assert(light:type() == tsd.LIGHT, "type should be LIGHT")
end)

test("ref:numParameters()", function()
  local geom = scene:createGeometry("sphere")
  geom:setParameter("radius", 1.0)
  assert(geom:numParameters() >= 1, "should have at least 1 parameter")
end)

test("ref:getParameter()", function()
  local geom = scene:createGeometry("sphere")
  geom:setParameter("radius", 1.5)
  local val = geom:getParameter("radius")
  assert(val ~= nil, "getParameter should return value")
  assert(math.abs(val - 1.5) < 0.001, "radius should be 1.5")
end)

-- Parameter removal
section("Parameter Removal")

test("ref:removeParameter(name)", function()
  local geom = scene:createGeometry("sphere")
  geom:setParameter("radius", 1.0)
  assert(geom:numParameters() >= 1, "should have parameter")
  geom:removeParameter("radius")
  local val = geom:getParameter("radius")
  assert(val == nil, "removed parameter should be nil")
end)

test("ref:removeAllParameters()", function()
  local mat = scene:createMaterial("matte")
  mat:setParameter("color", {1, 0, 0})
  mat:setParameter("opacity", 0.5)
  assert(mat:numParameters() >= 2, "should have at least 2 parameters")
  mat:removeAllParameters()
  assert(mat:numParameters() == 0, "should have no parameters after removeAll")
end)

-- Metadata
section("Metadata")

test("ref:setMetadata / ref:getMetadata (string)", function()
  local geom = scene:createGeometry("sphere")
  geom:setMetadata("author", "test_script")
  local val = geom:getMetadata("author")
  assert(val == "test_script", "metadata string should match")
end)

test("ref:setMetadata / ref:getMetadata (number)", function()
  local geom = scene:createGeometry("sphere")
  geom:setMetadata("version", 42)
  local val = geom:getMetadata("version")
  assert(val == 42, "metadata number should match")
end)

test("ref:numMetadata / ref:getMetadataName", function()
  local geom = scene:createGeometry("sphere")
  geom:setMetadata("key1", "value1")
  geom:setMetadata("key2", 2)
  assert(geom:numMetadata() >= 2, "should have at least 2 metadata entries")
end)

test("ref:removeMetadata", function()
  local geom = scene:createGeometry("sphere")
  geom:setMetadata("temp", "data")
  geom:removeMetadata("temp")
  local val = geom:getMetadata("temp")
  assert(val == nil, "removed metadata should be nil")
end)

-- Surface creation
section("Surface Creation")

test("scene:createSurface(name, geom, mat)", function()
  local geom = scene:createGeometry("sphere")
  local mat = scene:createMaterial("matte")
  local surface = scene:createSurface("testSurface", geom, mat)
  assert(surface ~= nil, "surface should be created")
  assert(surface:valid(), "surface ref should be valid")
end)

-- Layer operations
section("Layer Operations")

test("scene:addLayer(name)", function()
  local layer = scene:addLayer("testLayer")
  assert(layer ~= nil, "layer should be created")
end)

test("scene:layer(name)", function()
  scene:addLayer("namedLayer")
  local layer = scene:layer("namedLayer")
  assert(layer ~= nil, "layer should be retrieved by name")
end)

test("scene:layer(index)", function()
  local layer = scene:layer(0)
  assert(layer ~= nil, "layer should be retrieved by index")
end)

test("scene:defaultLayer()", function()
  local layer = scene:defaultLayer()
  assert(layer ~= nil, "default layer should exist")
end)

test("scene:numberOfLayers()", function()
  local count = scene:numberOfLayers()
  assert(count >= 1, "should have at least one layer")
end)

test("scene:layerIsActive / scene:setLayerActive", function()
  scene:addLayer("toggleLayer")
  assert(scene:layerIsActive("toggleLayer"), "new layer should be active")
  scene:setLayerActive("toggleLayer", false)
  assert(not scene:layerIsActive("toggleLayer"), "layer should be inactive")
  scene:setLayerActive("toggleLayer", true)
  assert(scene:layerIsActive("toggleLayer"), "layer should be active again")
end)

-- Array creation
section("Array Operations")

test("scene:createArray('float', count)", function()
  local arr = scene:createArray("float", 10)
  assert(arr ~= nil, "float array should be created")
  assert(arr:valid(), "array ref should be valid")
end)

test("scene:createArray('float', items0, items1)", function()
  local arr = scene:createArray("float", 4, 3)
  assert(arr ~= nil, "2D float array should be created")
  assert(arr:dim(0) == 4, "dim(0) should be 4")
  assert(arr:dim(1) == 3, "dim(1) should be 3")
  assert(arr:dim(2) == 1, "dim(2) should be 1 for 2D arrays")
  assert(arr:size() == 12, "size should be items0 * items1")
end)

test("scene:createArray('float', items0, items1, items2)", function()
  local arr = scene:createArray("float", 2, 3, 4)
  assert(arr ~= nil, "3D float array should be created")
  assert(arr:dim(0) == 2, "dim(0) should be 2")
  assert(arr:dim(1) == 3, "dim(1) should be 3")
  assert(arr:dim(2) == 4, "dim(2) should be 4")
  assert(arr:size() == 24, "size should be items0 * items1 * items2")
end)

test("scene:createArray('float3', count)", function()
  local arr = scene:createArray("float3", 5)
  assert(arr ~= nil, "float3 array should be created")
end)

test("array:setData with table", function()
  local arr = scene:createArray("float3", 3)
  arr:setData({
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0}
  })
end)

test("array:setData accepts float3 userdata entries", function()
  local arr = scene:createArray("float3", 2)
  arr:setData({
    tsd.float3(0.0, 1.0, 2.0),
    tsd.float3(3.0, 4.0, 5.0),
  })
  local data = arr:getData()
  assert(math.abs(data[1][1] - 0.0) < 0.001, "data[1][1] should match")
  assert(math.abs(data[2][3] - 5.0) < 0.001, "data[2][3] should match")
end)

test("array:setData accepts linear data for 2D arrays", function()
  local arr = scene:createArray("float3", 2, 2)
  arr:setData({
    {0, 0, 1},
    {0, 1, 0},
    {0, 1, 1},
    {1, 0, 0}
  })
  local data = arr:getData()
  assert(#data == 4, "2D array getData should return linear layout")
  assert(math.abs(data[1][3] - 1.0) < 0.001, "first vector should match")
  assert(math.abs(data[4][1] - 1.0) < 0.001, "last vector should match")
end)

test("array:setData accepts nested data for 2D arrays", function()
  local arr = scene:createArray("float3", 2, 2)
  arr:setData({
    {
      {0, 0, 1},
      {0, 1, 0},
    },
    {
      {0, 1, 1},
      {1, 0, 0},
    }
  })
  local data = arr:getData()
  assert(#data == 4, "2D nested setData should flatten to 4 values")
  assert(math.abs(data[2][2] - 1.0) < 0.001, "second vector should match")
  assert(math.abs(data[3][3] - 1.0) < 0.001, "third vector should match")
end)

test("array:setData accepts nested data for 3D arrays", function()
  local arr = scene:createArray("float", 2, 2, 2)
  arr:setData({
    {
      {1.0, 2.0},
      {3.0, 4.0},
    },
    {
      {5.0, 6.0},
      {7.0, 8.0},
    }
  })
  local data = arr:getData()
  assert(#data == 8, "3D nested setData should flatten to 8 values")
  assert(math.abs(data[1] - 1.0) < 0.001, "first scalar should match")
  assert(math.abs(data[8] - 8.0) < 0.001, "last scalar should match")
end)

test("array:getData roundtrip (float)", function()
  local arr = scene:createArray("float", 4)
  arr:setData({0.5, 1.5, 2.5, 3.5})
  local data = arr:getData()
  assert(#data == 4, "data table should have 4 elements")
  assert(math.abs(data[1] - 0.5) < 0.001, "data[1] should match")
  assert(math.abs(data[4] - 3.5) < 0.001, "data[4] should match")
end)

test("array:getData roundtrip (float3)", function()
  local arr = scene:createArray("float3", 2)
  arr:setData({
    {0.0, 1.0, 2.0},
    {3.0, 4.0, 5.0},
  })
  local data = arr:getData()
  assert(#data == 2, "data table should have 2 elements")
  assert(#data[1] == 3, "float3 entries should have 3 components")
  assert(math.abs(data[1][2] - 1.0) < 0.001, "data[1][2] should match")
  assert(math.abs(data[2][3] - 5.0) < 0.001, "data[2][3] should match")
end)

test("array:setData truncates oversized input", function()
  local arr = scene:createArray("float", 3)
  arr:setData({1.0, 2.0, 3.0, 4.0, 5.0})
  local data = arr:getData()
  assert(#data == 3, "data table should have 3 elements")
  assert(math.abs(data[3] - 3.0) < 0.001, "last stored value should be 3.0")
end)

test("array:setData leaves tail unchanged on short input", function()
  local arr = scene:createArray("float", 4)
  arr:setData({1.0, 2.0, 3.0, 4.0})
  arr:setData({9.0, 8.0})
  local data = arr:getData()
  assert(math.abs(data[1] - 9.0) < 0.001, "first element should be updated")
  assert(math.abs(data[2] - 8.0) < 0.001, "second element should be updated")
  assert(math.abs(data[3] - 3.0) < 0.001, "third element should be unchanged")
  assert(math.abs(data[4] - 4.0) < 0.001, "fourth element should be unchanged")
end)

test("array:size()", function()
  local arr = scene:createArray("float", 10)
  assert(arr:size() == 10, "array size should be 10")
end)

test("array:isEmpty()", function()
  local arr = scene:createArray("float", 5)
  assert(not arr:isEmpty(), "non-empty array should not be empty")
end)

test("ref:setParameterArray with inferred 1D dims", function()
  local geom = scene:createGeometry("triangle")
  local arr = geom:setParameterArray("vertex.position", "float3", {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
  })
  assert(arr ~= nil and arr:valid(), "returned array should be valid")
  assert(arr:size() == 3, "array should have 3 elements")
  local p = geom:parameter("vertex.position")
  assert(p ~= nil, "vertex.position parameter should be present")
end)

test("ref:setParameterArray with explicit 2D dims and linear data", function()
  local field = scene:createSpatialField("structuredRegular")
  local arr = field:setParameterArray("data", "float", 2, 2, {0, 1, 3, 3})
  assert(arr ~= nil and arr:valid(), "returned array should be valid")
  assert(arr:dim(0) == 2 and arr:dim(1) == 2 and arr:dim(2) == 1,
      "array dims should be 2x2")
  local data = arr:getData()
  assert(#data == 4, "array should contain 4 values")
  assert(math.abs(data[4] - 3.0) < 0.001, "last value should match input")
end)

test("ref:setParameterArray with explicit 2D dims rejects wrong element shape", function()
  local geom = scene:createGeometry("triangle")
  local ok, err = pcall(function()
    geom:setParameterArray("vertex.position", "float3", 2, 2, {0, 1, 3, 3})
  end)
  assert(not ok, "float3 array with scalar elements should fail")
  assert(err ~= nil, "error message should be present")
end)

test("ref:setParameterArray with inferred 3D dims from nested data", function()
  local field = scene:createSpatialField("structuredRegular")
  local arr = field:setParameterArray("data", "float", {
    {
      {1.0, 2.0},
      {3.0, 4.0},
    },
    {
      {5.0, 6.0},
      {7.0, 8.0},
    }
  })
  assert(arr ~= nil and arr:valid(), "returned array should be valid")
  assert(arr:dim(0) == 2 and arr:dim(1) == 2 and arr:dim(2) == 2,
      "array dims should be inferred as 2x2x2")
end)

-- Object access
section("Object Access")

test("scene:getGeometry(index)", function()
  scene:createGeometry("sphere")
  local geom = scene:getGeometry(0)
  assert(geom ~= nil, "geometry should be retrieved")
  assert(geom:valid(), "retrieved ref should be valid")
end)

test("scene:getMaterial(index)", function()
  scene:createMaterial("matte")
  local mat = scene:getMaterial(0)
  assert(mat ~= nil, "material should be retrieved")
end)

test("scene:getLight(index)", function()
  scene:createLight("point")
  local light = scene:getLight(0)
  assert(light ~= nil, "light should be retrieved")
end)

-- Iteration
section("Iteration")

test("scene:forEachGeometry", function()
  local count = 0
  scene:forEachGeometry(function(g)
    count = count + 1
  end)
  assert(count >= 0, "forEachGeometry should iterate")
end)

test("scene:forEachMaterial", function()
  local count = 0
  scene:forEachMaterial(function(m)
    count = count + 1
  end)
  assert(count >= 0, "forEachMaterial should iterate")
end)

test("scene:forEachLight", function()
  local count = 0
  scene:forEachLight(function(l)
    count = count + 1
  end)
  assert(count >= 0, "forEachLight should iterate")
end)

test("scene:forEachSurface", function()
  local count = 0
  scene:forEachSurface(function(s)
    count = count + 1
  end)
  assert(count >= 0, "forEachSurface should iterate")
end)

-- Scene removal operations
section("Scene Removal Operations")

test("scene:removeUnusedObjects", function()
  scene:removeUnusedObjects()
  -- No error means success
end)

test("scene:cleanupScene", function()
  scene:cleanupScene()
end)

-- Animation API
section("Animation API")

test("scene:numberOfAnimations()", function()
  local count = scene:numberOfAnimations()
  assert(count >= 0, "animation count should be non-negative")
end)

test("scene:addAnimation(name)", function()
  local anim = scene:addAnimation("testAnim")
  assert(anim ~= nil, "animation should be created")
  assert(scene:numberOfAnimations() >= 1, "should have at least 1 animation")
end)

test("animation name", function()
  local anim = scene:addAnimation("myAnimation")
  assert(anim.name == "myAnimation", "animation name should match")
  anim.name = "renamedAnimation"
  assert(anim.name == "renamedAnimation", "renamed animation name should match")
end)

test("scene:setAnimationTime / getAnimationTime", function()
  scene:setAnimationTime(0.5)
  local t = scene:getAnimationTime()
  assert(math.abs(t - 0.5) < 0.001, "animation time should be 0.5")
end)

test("scene:setAnimationIncrement / getAnimationIncrement", function()
  scene:setAnimationIncrement(0.02)
  local inc = scene:getAnimationIncrement()
  assert(math.abs(inc - 0.02) < 0.001, "animation increment should be 0.02")
end)

test("scene:incrementAnimationTime", function()
  scene:setAnimationTime(0.0)
  scene:setAnimationIncrement(0.1)
  scene:incrementAnimationTime()
  local t = scene:getAnimationTime()
  assert(math.abs(t - 0.1) < 0.001, "animation time should be 0.1 after increment")
end)

-- IO/Procedural generators
section("Procedural Generators")

test("tsd.io module and procedural generators exist", function()
  assert(tsd.io ~= nil, "tsd.io module should exist")
  local generators = {
    "generateRandomSpheres", "generateMonkey", "generateCylinders",
    "generateMaterialOrb", "generateDefaultLights", "generateHdriDome",
    "generateRtow",
  }
  for _, name in ipairs(generators) do
    assert(tsd.io[name] ~= nil, name .. " should exist")
  end
end)

-- Import functions existence check
section("Import Functions")

test("tsd.io import functions exist", function()
  local importers = { "importOBJ", "importGLTF", "importPLY", "importHDRI" }
  for _, name in ipairs(importers) do
    assert(tsd.io[name] ~= nil, name .. " should exist")
  end
end)

-- Complete workflow test
section("Complete Workflow")

test("Create complete scene with sphere", function()
  local sphereGeom = scene:createGeometry("sphere")
  sphereGeom:setParameter("radius", 0.5)

  local redMat = scene:createMaterial("matte")
  redMat:setParameter("color", tsd.float3(0.8, 0.1, 0.1))

  local surface = scene:createSurface("redSphere", sphereGeom, redMat)
  assert(surface:valid(), "surface should be valid")

  local light = scene:createLight("directional")
  light:setParameter("direction", tsd.float3(-1, -1, -0.5))
  light:setParameter("irradiance", 5.0)
end)

test("Create scene with transform", function()
  local geom = scene:createGeometry("sphere")
  geom:setParameter("radius", 1.0)

  local mat = scene:createMaterial("physicallyBased")
  mat:setParameter("baseColor", tsd.float3(0.2, 0.5, 0.8))
  mat:setParameter("metallic", 0.5)
  mat:setParameter("roughness", 0.3)

  local surface = scene:createSurface("pbrSphere", geom, mat)

  local xfm = tsd.translation(tsd.float3(2.0, 0.0, 0.0)) * tsd.scaling(0.5)
  assert(xfm ~= nil, "transform should be created")
end)

test("Create triangle with vertex arrays", function()
  local geom = scene:createGeometry("triangle")
  geom.name = "DirectTriangle"

  local posArray = scene:createArray("float3", 3)
  posArray:setData({
    {-0.5, -0.5, 0.0},
    { 0.5, -0.5, 0.0},
    { 0.0,  0.5, 0.0}
  })
  assert(posArray:size() == 3, "array should have 3 elements")

  geom:setParameter("vertex.position", posArray)

  local mat = scene:createMaterial("matte")
  mat:setParameter("color", {0.9, 0.9, 0.9})
  local surf = scene:createSurface("DirectTriangleSurface", geom, mat)
  assert(surf:valid(), "surface should be valid")
end)

-- Inline parameter table on creation methods
section("Inline Parameter Tables")

test("createGeometry with param table", function()
  local geom = scene:createGeometry("sphere", { radius = 2.5 })
  assert(geom:valid(), "geometry should be valid")
  local r = geom:getParameter("radius")
  assert(r ~= nil, "radius should be set")
  assert(math.abs(r - 2.5) < 0.001, "radius should be 2.5, got " .. tostring(r))
end)

test("createMaterial with param table", function()
  local mat = scene:createMaterial("matte", { color = tsd.float3(1.0, 0.0, 0.0) })
  assert(mat:valid(), "material should be valid")
  local c = mat:getParameter("color")
  assert(c ~= nil, "color should be set")
  assert(math.abs(c.x - 1.0) < 0.001, "color.x should be 1.0")
end)

test("createLight with param table", function()
  local light = scene:createLight("directional", {
    direction = tsd.float3(-1, -1, -0.5),
    irradiance = 3.0,
  })
  assert(light:valid(), "light should be valid")
  local irr = light:getParameter("irradiance")
  assert(irr ~= nil, "irradiance should be set")
  assert(math.abs(irr - 3.0) < 0.001, "irradiance should be 3.0")
end)

test("createCamera with param table", function()
  local cam = scene:createCamera("perspective", { fovy = 1.2 })
  assert(cam:valid(), "camera should be valid")
  local f = cam:getParameter("fovy")
  assert(f ~= nil, "fovy should be set")
  assert(math.abs(f - 1.2) < 0.001, "fovy should be 1.2")
end)

test("createSampler with param table", function()
  local sampler = scene:createSampler("image2D", { filter = "nearest" })
  assert(sampler:valid(), "sampler should be valid")
  assert(sampler:numParameters() >= 1, "should have at least 1 parameter")
end)

test("createVolume with param table", function()
  local vol = scene:createVolume("transferFunction1D", { unitDistance = 0.5 })
  assert(vol:valid(), "volume should be valid")
  local ud = vol:getParameter("unitDistance")
  assert(ud ~= nil, "unitDistance should be set")
  assert(math.abs(ud - 0.5) < 0.001, "unitDistance should be 0.5")
end)

test("createSpatialField with param table", function()
  local field = scene:createSpatialField("structuredRegular", {
    filter = "nearest",
  })
  assert(field:valid(), "spatial field should be valid")
  assert(field:numParameters() >= 1, "should have at least 1 parameter")
end)

test("createSurface with param table", function()
  local geom = scene:createGeometry("sphere")
  local mat = scene:createMaterial("matte")
  local surf = scene:createSurface("paramSurf", geom, mat, { id = 42 })
  assert(surf:valid(), "surface should be valid")
  local id = surf:getParameter("id")
  assert(id ~= nil, "id should be set")
  assert(id == 42, "id should be 42")
end)

test("createGeometry without param table (backward compat)", function()
  local geom = scene:createGeometry("sphere")
  assert(geom:valid(), "geometry should be valid without params")
end)

test("param table with plain-table-as-vector (float3)", function()
  local mat = scene:createMaterial("matte", { color = {0.5, 0.5, 0.5} })
  assert(mat:valid(), "material should be valid")
  local c = mat:getParameter("color")
  assert(c ~= nil, "color should be set")
  assert(math.abs(c.x - 0.5) < 0.001, "color.x should be 0.5")
  assert(math.abs(c.y - 0.5) < 0.001, "color.y should be 0.5")
  assert(math.abs(c.z - 0.5) < 0.001, "color.z should be 0.5")
end)

test("param table with multiple params", function()
  local mat = scene:createMaterial("physicallyBased", {
    baseColor = tsd.float3(0.2, 0.5, 0.8),
    metallic = 0.5,
    roughness = 0.3,
  })
  assert(mat:valid(), "material should be valid")
  local m = mat:getParameter("metallic")
  assert(m ~= nil, "metallic should be set")
  assert(math.abs(m - 0.5) < 0.001, "metallic should be 0.5")
  local r = mat:getParameter("roughness")
  assert(r ~= nil, "roughness should be set")
  assert(math.abs(r - 0.3) < 0.001, "roughness should be 0.3")
end)

-- Token type
section("Token Type")

test("Token() default constructor", function()
  local t = tsd.Token.new()
  assert(t ~= nil, "Token should be created")
  assert(t:empty(), "default Token should be empty")
end)

test("Token(str) and str()", function()
  local t = tsd.Token.new("hello")
  assert(not t:empty(), "Token('hello') should not be empty")
  assert(t:str() == "hello", "str() should return 'hello'")
end)

test("Token tostring", function()
  local t = tsd.Token.new("world")
  assert(tostring(t) == "world", "tostring should return 'world'")
end)

-- Scene creation
section("Scene Creation")

test("tsd.createScene()", function()
  local s = tsd.createScene()
  assert(s ~= nil, "createScene should return a scene")
end)

-- Type constants
section("Type Constants")

test("ANARI type constants exist and are unique", function()
  local names = {
    "GEOMETRY", "MATERIAL", "LIGHT", "CAMERA",
    "SURFACE", "VOLUME", "SAMPLER", "ARRAY", "SPATIAL_FIELD"
  }
  for _, name in ipairs(names) do
    assert(tsd[name] ~= nil, "tsd." .. name .. " should exist")
  end
  -- Verify they can be used with numberOfObjects (functional test)
  local s = tsd.createScene()
  local before = s:numberOfObjects(tsd.GEOMETRY)
  s:createGeometry("sphere")
  local after = s:numberOfObjects(tsd.GEOMETRY)
  assert(after == before + 1, "GEOMETRY constant should work with numberOfObjects")
end)

-- Object methods
section("Object Methods (Extended)")

test("ref:index() returns integer", function()
  local geom = scene:createGeometry("sphere")
  local idx = geom:index()
  assert(type(idx) == "number", "index() should return a number")
  assert(idx >= 0, "index() should be non-negative")
end)

test("ref:parameterNameAt(i)", function()
  local geom = scene:createGeometry("sphere")
  geom:setParameter("radius", 1.0)
  local name = geom:parameterNameAt(0)
  assert(type(name) == "string", "parameterNameAt should return a string")
  assert(name == "radius", "parameterNameAt(0) should be 'radius'")
end)

-- Surface/Volume accessors
section("Surface/Volume Accessors")

test("Surface:geometry() and Surface:material()", function()
  local geom = scene:createGeometry("sphere")
  local mat = scene:createMaterial("matte")
  local surf = scene:createSurface("accessorSurf", geom, mat)
  local g = surf:geometry()
  assert(g ~= nil, "geometry() should return non-nil")
  local m = surf:material()
  assert(m ~= nil, "material() should return non-nil")
end)

test("Volume:spatialField()", function()
  local field = scene:createSpatialField("structuredRegular")
  local vol = scene:createVolume("transferFunction1D")
  vol:setParameter("value", field)
  local sf = vol:spatialField()
  assert(sf ~= nil, "spatialField() should return non-nil")
end)

test("SpatialField:computeValueRange()", function()
  local field = scene:createSpatialField("structuredRegular")
  field:setParameterArray("data", "float", 2, 2, 2, {0, 1, 2, 3, 4, 5, 6, 7})
  local range = field:computeValueRange()
  assert(range ~= nil, "computeValueRange should return non-nil")
  assert(range.x <= range.y, "min should be <= max")
end)

-- Vector arithmetic
section("Vector Arithmetic")

test("float3 addition", function()
  local a = tsd.float3(1, 2, 3)
  local b = tsd.float3(4, 5, 6)
  local c = a + b
  assert(math.abs(c.x - 5) < 0.001 and math.abs(c.y - 7) < 0.001 and math.abs(c.z - 9) < 0.001,
    "float3 addition should work")
end)

test("float3 subtraction", function()
  local a = tsd.float3(4, 5, 6)
  local b = tsd.float3(1, 2, 3)
  local c = a - b
  assert(math.abs(c.x - 3) < 0.001 and math.abs(c.y - 3) < 0.001 and math.abs(c.z - 3) < 0.001,
    "float3 subtraction should work")
end)

test("float3 scalar multiply", function()
  local a = tsd.float3(1, 2, 3)
  local c = a * 2
  assert(math.abs(c.x - 2) < 0.001 and math.abs(c.y - 4) < 0.001 and math.abs(c.z - 6) < 0.001,
    "float3 scalar multiply should work")
end)

test("float3 scalar divide", function()
  local a = tsd.float3(2, 4, 6)
  local c = a / 2
  assert(math.abs(c.x - 1) < 0.001 and math.abs(c.y - 2) < 0.001 and math.abs(c.z - 3) < 0.001,
    "float3 scalar divide should work")
end)

test("float2 arithmetic", function()
  local a = tsd.float2(1, 2)
  local b = tsd.float2(3, 4)
  local c = a + b
  assert(math.abs(c.x - 4) < 0.001 and math.abs(c.y - 6) < 0.001,
    "float2 addition should work")
end)

test("float4 arithmetic", function()
  local a = tsd.float4(1, 2, 3, 4)
  local b = tsd.float4(5, 6, 7, 8)
  local c = a + b
  assert(math.abs(c.x - 6) < 0.001 and math.abs(c.w - 12) < 0.001,
    "float4 addition should work")
end)

-- matrix constructors
section("Matrix Constructors")

test("mat3() constructor", function()
  local m = tsd.mat3()
  assert(m ~= nil, "mat3() should construct")
end)

test("mat3(float3, float3, float3) constructor", function()
  local c0 = tsd.float3(1.0, 2.0, 3.0)
  local c1 = tsd.float3(4.0, 5.0, 6.0)
  local c2 = tsd.float3(7.0, 8.0, 9.0)
  local m = tsd.mat3(c0, c1, c2)
  assert(m ~= nil, "mat3 columns constructor should construct")
  assert(m[0].x == 1.0 and m[1].y == 5.0 and m[2].z == 9.0,
      "mat3 columns should map as expected")
end)

test("mat4() constructor", function()
  local m = tsd.mat4()
  assert(m ~= nil, "mat4() should construct")
end)

test("mat4(float4, float4, float4, float4) constructor", function()
  local c0 = tsd.float4(1.0, 0.0, 0.0, 0.0)
  local c1 = tsd.float4(0.0, 1.0, 0.0, 0.0)
  local c2 = tsd.float4(0.0, 0.0, 1.0, 0.0)
  local c3 = tsd.float4(10.0, 20.0, 30.0, 1.0)
  local m = tsd.mat4(c0, c1, c2, c3)
  assert(m ~= nil, "mat4 columns constructor should construct")
  assert(m[3].x == 10.0 and m[3].y == 20.0 and m[3].z == 30.0 and m[3].w == 1.0,
      "mat4 columns should map as expected")
end)

test("mat4.identity exists", function()
  local m = tsd.mat4.identity
  assert(m ~= nil, "mat4.identity should exist")
end)

-- Scene methods (extended)
section("Scene Methods (Extended)")

test("scene:numberOfObjects(type)", function()
  local s = tsd.createScene()
  s:createGeometry("sphere")
  s:createGeometry("sphere")
  local n = s:numberOfObjects(tsd.GEOMETRY)
  assert(n == 2, "should have 2 geometries, got " .. tostring(n))
end)

test("scene:defaultMaterial()", function()
  local mat = scene:defaultMaterial()
  assert(mat ~= nil, "defaultMaterial should return non-nil")
end)

test("scene:setAllLayersActive() and numberOfActiveLayers()", function()
  scene:setAllLayersActive()
  local n = scene:numberOfActiveLayers()
  assert(n >= 1, "should have at least 1 active layer")
end)

test("scene:removeAllObjects()", function()
  local s = tsd.createScene()
  s:createGeometry("sphere")
  s:createMaterial("matte")
  s:removeAllObjects()
  assert(s:numberOfObjects(tsd.GEOMETRY) == 0, "should have 0 geometries")
  assert(s:numberOfObjects(tsd.MATERIAL) == 0, "should have 0 materials")
end)

section("Scene Object Lifecycle")

test("scene:create* inserts and scene:removeObject removes objects per type", function()
  local cases = {
    {
      name = "GEOMETRY",
      count = function(s)
        local n = 0
        s:forEachGeometry(function(_) n = n + 1 end)
        return n
      end,
      create = function(s)
        return s:createGeometry("sphere")
      end
    },
    {
      name = "MATERIAL",
      count = function(s)
        local n = 0
        s:forEachMaterial(function(_) n = n + 1 end)
        return n
      end,
      create = function(s)
        return s:createMaterial("matte")
      end
    },
    {
      name = "LIGHT",
      count = function(s)
        local n = 0
        s:forEachLight(function(_) n = n + 1 end)
        return n
      end,
      create = function(s)
        return s:createLight("point")
      end
    },
    {
      name = "CAMERA",
      count = function(s)
        local n = 0
        s:forEachCamera(function(_) n = n + 1 end)
        return n
      end,
      create = function(s)
        return s:createCamera("perspective")
      end
    },
    {
      name = "SURFACE",
      count = function(s)
        local n = 0
        s:forEachSurface(function(_) n = n + 1 end)
        return n
      end,
      create = function(s)
        local geom = s:createGeometry("sphere")
        local mat = s:createMaterial("matte")
        return s:createSurface("lifecycleSurface", geom, mat)
      end
    },
    {
      name = "VOLUME",
      count = function(s)
        local n = 0
        s:forEachVolume(function(_) n = n + 1 end)
        return n
      end,
      create = function(s)
        return s:createVolume("transferFunction1D")
      end
    },
    {
      name = "SAMPLER",
      count = function(s)
        local n = 0
        s:forEachSampler(function(_) n = n + 1 end)
        return n
      end,
      create = function(s)
        return s:createSampler("image2D")
      end
    },
    {
      name = "ARRAY",
      count = function(s)
        local n = 0
        s:forEachArray(function(_) n = n + 1 end)
        return n
      end,
      create = function(s)
        return s:createArray("float", 4)
      end
    },
    {
      name = "SPATIAL_FIELD",
      count = function(s)
        local n = 0
        s:forEachSpatialField(function(_) n = n + 1 end)
        return n
      end,
      create = function(s)
        return s:createSpatialField("structuredRegular")
      end
    }
  }

  for _, case in ipairs(cases) do
    local s = tsd.createScene()
    local before = case.count(s)
    local obj = case.create(s)
    assert(obj ~= nil and obj:valid(), case.name .. " object should be valid")
    local afterCreate = case.count(s)
    assert(afterCreate == before + 1,
      case.name .. " count should increment by 1 after creation")
    s:removeObject(obj)
    local afterRemove = case.count(s)
    assert(afterRemove == before,
      case.name .. " count should return to baseline after removeObject")
  end
end)

test("scene:removeObject(nil) is a no-op", function()
  local s = tsd.createScene()
  local before = 0
  s:forEachGeometry(function(_) before = before + 1 end)
  s:removeObject(nil)
  local after = 0
  s:forEachGeometry(function(_) after = after + 1 end)
  assert(after == before, "removeObject(nil) should not change object counts")
end)

test("scene:removeNode(node) removes node but keeps referenced object", function()
  local s = tsd.createScene()
  local layer = s:defaultLayer()
  local root = layer:root()
  local geom = s:createGeometry("sphere")
  local mat = s:createMaterial("matte")
  local surf = s:createSurface("instancedKeepSurface", geom, mat)
  local surfaceCountBefore = 0
  s:forEachSurface(function(_) surfaceCountBefore = surfaceCountBefore + 1 end)
  local layerSizeBefore = layer:size()
  local surfNode = s:insertObjectNode(root, surf)
  assert(layer:size() == layerSizeBefore + 1, "surface node insertion should grow layer by 1")

  s:removeNode(surfNode)

  assert(layer:size() == layerSizeBefore, "removing instanced node should restore layer size")
  local surfaceCountAfter = 0
  s:forEachSurface(function(_) surfaceCountAfter = surfaceCountAfter + 1 end)
  assert(surfaceCountAfter == surfaceCountBefore,
    "removeNode(node) should keep referenced live surface object")
end)

test("scene:removeNode(node, true) removes node and referenced object", function()
  local s = tsd.createScene()
  local layer = s:defaultLayer()
  local root = layer:root()
  local geom = s:createGeometry("sphere")
  local mat = s:createMaterial("matte")
  local surf = s:createSurface("instancedDeleteSurface", geom, mat)
  local surfaceCountBefore = 0
  s:forEachSurface(function(_) surfaceCountBefore = surfaceCountBefore + 1 end)
  local layerSizeBefore = layer:size()
  local surfNode = s:insertObjectNode(root, surf)
  assert(layer:size() == layerSizeBefore + 1, "surface node insertion should grow layer by 1")

  s:removeNode(surfNode, true)

  assert(layer:size() == layerSizeBefore, "removing instanced node should restore layer size")
  local surfaceCountAfter = 0
  s:forEachSurface(function(_) surfaceCountAfter = surfaceCountAfter + 1 end)
  assert(surfaceCountAfter == surfaceCountBefore - 1,
    "removeNode(node, true) should remove referenced live surface object")
end)

test("scene:removeNode(root) is a no-op", function()
  local s = tsd.createScene()
  local layer = s:defaultLayer()
  local root = layer:root()
  local layerSizeBefore = layer:size()
  s:removeNode(root, true)
  assert(layer:size() == layerSizeBefore, "root node removal should be ignored")
end)

test("scene:removeNode(invalidNode) is a no-op", function()
  local s = tsd.createScene()
  local layer = s:defaultLayer()
  local root = layer:root()
  local invalid = root:child(999) -- out of range, returns invalid ref
  assert(not invalid:valid(), "child(999) should be invalid")
  local sizeBefore = layer:size()
  s:removeNode(invalid)
  s:removeNode(invalid, true)
  assert(layer:size() == sizeBefore, "invalid node removal should not change layer")
end)

test("scene:removeLayer(name)", function()
  local s = tsd.createScene()
  s:addLayer("tempLayer")
  local before = s:numberOfLayers()
  s:removeLayer("tempLayer")
  local after = s:numberOfLayers()
  assert(after == before - 1, "layer count should decrease by 1")
end)

test("scene:defragmentObjectStorage()", function()
  local s = tsd.createScene()
  s:createGeometry("sphere")
  s:defragmentObjectStorage()
  -- No error means success
end)

-- Layer tree operations
section("Layer Tree Operations")

test("Layer:root()", function()
  local layer = scene:defaultLayer()
  local root = layer:root()
  assert(root ~= nil, "root should exist")
  assert(root:valid(), "root should be valid")
  assert(root:isRoot(), "root should be root")
end)

test("Layer:size() and Layer:empty()", function()
  local s = tsd.createScene()
  local layer = s:addLayer("sizeTest")
  -- Insert a child so the layer is non-empty
  s:insertChildNode(layer:root(), "child")
  local sz = layer:size()
  assert(type(sz) == "number", "size should be a number")
  assert(sz >= 2, "layer should have root + child, got " .. tostring(sz))
  assert(not layer:empty(), "layer with nodes should not be empty")
end)

test("Layer:at(index)", function()
  local layer = scene:defaultLayer()
  local node = layer:at(0)
  assert(node ~= nil, "at(0) should return a node")
end)

test("Layer:foreach(fn)", function()
  local layer = scene:defaultLayer()
  local count = 0
  layer:foreach(function(node, level)
    count = count + 1
    assert(type(level) == "number", "level should be a number")
  end)
  assert(count >= 1, "foreach should visit at least root")
end)

test("Layer:foreach provides LayerNodeRef with full API", function()
  local s = tsd.createScene()
  local layer = s:addLayer("foreachRefTest")
  local root = layer:root()
  s:insertChildNode(root, "childA")
  s:insertChildTransformNode(root, tsd.mat4.identity, "childXfm")

  local names = {}
  layer:foreach(function(node, level)
    -- node should be a LayerNodeRef with name, valid, isTransform, etc.
    assert(node:valid(), "foreach node should be valid")
    assert(node.name ~= nil, "foreach node should have name property")
    table.insert(names, node.name)
  end)
  assert(#names >= 3, "should visit root + 2 children, got " .. #names)

  -- Verify name-based lookup works (the reported bug scenario)
  local found = nil
  layer:foreach(function(node, level)
    if node.name == "childA" then
      found = node
      return false
    end
  end)
  assert(found ~= nil, "should find node by name")
  assert(found:valid(), "found node should be valid")
  assert(found.name == "childA", "found node name should match")
  -- found should be usable where LayerNodeRef is expected
  local child2 = s:insertChildNode(found, "grandchild")
  assert(child2:valid(), "insertChildNode with foreach result should work")
end)

test("Layer:foreach visits all nodes", function()
  local layer = scene:defaultLayer()
  local count = 0
  layer:foreach(function(node, level)
    count = count + 1
  end)
  assert(count >= 1, "foreach should visit at least root")
end)

test("Layer:foreach early termination", function()
  local layer = scene:defaultLayer()
  local count = 0
  layer:foreach(function(node, level)
    count = count + 1
    return false -- stop after first
  end)
  assert(count == 1, "foreach should stop after returning false")
end)

-- LayerNode properties
section("LayerNode Properties")

test("insertChildNode and node properties", function()
  local layer = scene:defaultLayer()
  local root = layer:root()
  local child = scene:insertChildNode(root, "testChild")
  assert(child ~= nil, "child should be created")
  assert(child:valid(), "child should be valid")
  assert(child.name == "testChild", "child name should be 'testChild'")
  assert(not child:isRoot(), "child should not be root")
end)

test("LayerNode setEnabled/isEnabled", function()
  local layer = scene:defaultLayer()
  local root = layer:root()
  local child = scene:insertChildNode(root, "enableTest")
  assert(child:isEnabled(), "new node should be enabled")
  child:setEnabled(false)
  assert(not child:isEnabled(), "node should be disabled")
  child:setEnabled(true)
  assert(child:isEnabled(), "node should be re-enabled")
end)

test("LayerNode:child(index)", function()
  local s = tsd.createScene()
  local layer = s:addLayer("childIdxTest")
  local root = layer:root()
  local a = s:insertChildNode(root, "a")
  local b = s:insertChildNode(root, "b")
  local c = s:insertChildNode(root, "c")
  assert(root:child(0):valid(), "child(0) should be valid")
  assert(root:child(0).name == "a", "child(0) should be 'a'")
  assert(root:child(1).name == "b", "child(1) should be 'b'")
  assert(root:child(2).name == "c", "child(2) should be 'c'")
  assert(not root:child(3):valid(), "child(3) should be invalid (out of range)")
  assert(not root:child(-1):valid(), "child(-1) should be invalid")
end)

test("LayerNode:childByName(name)", function()
  local s = tsd.createScene()
  local layer = s:addLayer("childNameTest")
  local root = layer:root()
  s:insertChildNode(root, "alpha")
  s:insertChildNode(root, "beta")
  s:insertChildNode(root, "gamma")
  local found = root:childByName("beta")
  assert(found:valid(), "childByName should find 'beta'")
  assert(found.name == "beta", "found node name should be 'beta'")
  local missing = root:childByName("nonexistent")
  assert(not missing:valid(), "childByName for missing name should be invalid")
end)

test("insertChildTransformNode", function()
  local layer = scene:defaultLayer()
  local root = layer:root()
  local xfm = tsd.translation(tsd.float3(1, 2, 3))
  local node = scene:insertChildTransformNode(root, xfm, "xfmNode")
  assert(node ~= nil, "transform node should be created")
  assert(node:valid(), "transform node should be valid")
  assert(node:isTransform(), "node should be a transform")
end)

test("insertObjectNode", function()
  local layer = scene:defaultLayer()
  local root = layer:root()
  local geom = scene:createGeometry("sphere")
  local mat = scene:createMaterial("matte")
  local surf = scene:createSurface("nodeSurf", geom, mat)
  local surfNode = scene:insertObjectNode(root, surf)
  assert(surfNode ~= nil and surfNode:valid(), "surface node should be valid")
  assert(surfNode:isObject(), "surface node should be an object node")

  local light = scene:createLight("point")
  local lightNode = scene:insertObjectNode(root, light)
  assert(lightNode ~= nil and lightNode:valid(), "light node should be valid")

  local vol = scene:createVolume("transferFunction1D")
  local volNode = scene:insertObjectNode(root, vol)
  assert(volNode ~= nil and volNode:valid(), "volume node should be valid")
end)

-- Animation (extended)
section("Animation (Extended)")

test("animation:info()", function()
  local anim = scene:addAnimation("infoAnim")
  local info = anim:info()
  assert(type(info) == "string", "info() should return a string")
end)

test("animation:timeStepCount()", function()
  local anim = scene:addAnimation("tsAnim")
  local count = anim:timeStepCount()
  assert(type(count) == "number", "timeStepCount() should return a number")
  assert(count >= 0, "timeStepCount should be non-negative")
end)

test("animation:update(time)", function()
  local anim = scene:addAnimation("updateAnim")
  anim:update(0.5) -- no error means success
end)

test("scene:removeAnimation(anim)", function()
  local s = tsd.createScene()
  local anim = s:addAnimation("toRemove")
  assert(s:numberOfAnimations() == 1, "should have 1 animation")
  s:removeAnimation(anim)
  assert(s:numberOfAnimations() == 0, "should have 0 animations after removal")
end)

test("scene:removeAllAnimations()", function()
  local s = tsd.createScene()
  s:addAnimation("anim1")
  s:addAnimation("anim2")
  assert(s:numberOfAnimations() == 2, "should have 2 animations")
  s:removeAllAnimations()
  assert(s:numberOfAnimations() == 0, "should have 0 animations after removeAll")
end)

-- ForEach (extended)
section("ForEach (Extended)")

test("scene:forEachCamera", function()
  local s = tsd.createScene()
  s:createCamera("perspective")
  local count = 0
  s:forEachCamera(function(c)
    count = count + 1
  end)
  assert(count == 1, "should iterate over 1 camera")
end)

test("scene:forEachVolume", function()
  local s = tsd.createScene()
  s:createVolume("transferFunction1D")
  local count = 0
  s:forEachVolume(function(v)
    count = count + 1
  end)
  assert(count == 1, "should iterate over 1 volume")
end)

test("scene:forEachSpatialField", function()
  local s = tsd.createScene()
  s:createSpatialField("structuredRegular")
  local count = 0
  s:forEachSpatialField(function(f)
    count = count + 1
  end)
  assert(count == 1, "should iterate over 1 spatial field")
end)

test("scene:forEachSampler", function()
  local s = tsd.createScene()
  s:createSampler("image2D")
  local count = 0
  s:forEachSampler(function(s)
    count = count + 1
  end)
  assert(count == 1, "should iterate over 1 sampler")
end)

test("scene:forEachArray", function()
  local s = tsd.createScene()
  s:createArray("float", 5)
  local count = 0
  s:forEachArray(function(a)
    count = count + 1
  end)
  assert(count == 1, "should iterate over 1 array")
end)

test("forEach early termination with return false", function()
  local s = tsd.createScene()
  s:createGeometry("sphere")
  s:createGeometry("sphere")
  s:createGeometry("sphere")
  local count = 0
  s:forEachGeometry(function(g)
    count = count + 1
    return false -- stop after first
  end)
  assert(count == 1, "forEach should stop after returning false")
end)

-- Array types (extended)
section("Array Types (Extended)")

test("int array creation and data", function()
  local arr = scene:createArray("int", 3)
  assert(arr:valid(), "int array should be valid")
  arr:setData({10, 20, 30})
  local data = arr:getData()
  assert(data[1] == 10 and data[2] == 20 and data[3] == 30,
    "int array data should roundtrip")
end)

test("uint array creation and data", function()
  local arr = scene:createArray("uint", 3)
  assert(arr:valid(), "uint array should be valid")
  arr:setData({100, 200, 300})
  local data = arr:getData()
  assert(data[1] == 100 and data[3] == 300, "uint array data should roundtrip")
end)

test("array:elementType() and elementSize()", function()
  local arr = scene:createArray("float3", 4)
  local et = arr:elementType()
  assert(et ~= nil, "elementType() should return non-nil")
  local es = arr:elementSize()
  assert(es ~= nil, "elementSize() should return non-nil")
  -- float3 = 3 * 4 bytes = 12
  assert(tonumber(es) == 12, "float3 elementSize should be 12 bytes")
end)

-- Metadata (extended)
section("Metadata (Extended)")

test("boolean metadata values", function()
  local geom = scene:createGeometry("sphere")
  geom:setMetadata("visible", true)
  local val = geom:getMetadata("visible")
  assert(val == true, "boolean metadata should roundtrip as true")
  geom:setMetadata("visible", false)
  val = geom:getMetadata("visible")
  assert(val == false, "boolean metadata should roundtrip as false")
end)

-- IO functions (extended)
section("IO Functions (Extended)")

test("tsd.io.generateSphereSetVolume exists", function()
  assert(tsd.io.generateSphereSetVolume ~= nil, "generateSphereSetVolume should exist")
end)

test("remaining io functions exist", function()
  local funcs = {
    "importUSD", "importPDB", "importSWC", "importVolume",
    "importRAW", "importNVDB", "importMHD",
    "saveScene", "loadScene",
  }
  for _, name in ipairs(funcs) do
    assert(tsd.io[name] ~= nil, name .. " should exist")
  end
end)

-- Render safety guards
section("Render Safety")

test("tsd.render.createRenderIndex rejects nil device", function()
  local ok, err = pcall(function()
    tsd.render.createRenderIndex(scene, nil)
  end)
  assert(not ok, "createRenderIndex(nil) should fail")
  assert(err ~= nil, "error message should be present")
end)

test("tsd.render.getWorldBounds rejects nil handles", function()
  local ok, err = pcall(function()
    tsd.render.getWorldBounds(nil, nil)
  end)
  assert(not ok, "getWorldBounds(nil, nil) should fail")
  assert(err ~= nil, "error message should be present")
end)

test("tsd.render.renderToFile rejects nil pipeline", function()
  local ok, err = pcall(function()
    tsd.render.renderToFile(nil, 1, "out.ppm", 64, 64)
  end)
  assert(not ok, "renderToFile(nil, ...) should fail")
  assert(err ~= nil, "error message should be present")
end)

-- Verify tostring output (Ref types should print without "Ref" suffix)
section("Tostring Output")

test("Geometry ref tostring has no 'Ref' suffix", function()
  local geom = scene:createGeometry("sphere")
  local s = tostring(geom)
  assert(s:match("^Geometry%("), "tostring should start with 'Geometry(', got: " .. s)
  assert(not s:match("Ref"), "tostring should not contain 'Ref', got: " .. s)
end)

test("Material ref tostring has no 'Ref' suffix", function()
  local mat = scene:createMaterial("matte")
  local s = tostring(mat)
  assert(s:match("^Material%("), "tostring should start with 'Material(', got: " .. s)
end)

test("LayerNode ref tostring has no 'Ref' suffix", function()
  local layer = scene:defaultLayer()
  local root = layer:root()
  local child = scene:insertChildNode(root, "tsNode")
  local s = tostring(child)
  assert(s:match("^LayerNode%("), "tostring should start with 'LayerNode(', got: " .. s)
  assert(not s:match("Ref"), "tostring should not contain 'Ref', got: " .. s)
end)

-- Print summary
section("SUMMARY")
print(string.format("Passed: %d", passed))
print(string.format("Failed: %d", failed))

if #errors > 0 then
  print("\nFailed tests:")
  for _, e in ipairs(errors) do
    print("  - " .. e.name .. ": " .. e.error)
  end
end

print(string.format("\nTotal: %d/%d tests passed", passed, passed + failed))
