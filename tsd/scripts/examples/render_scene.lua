-- Render RTOW + HDRI Example
-- Sequence:
--   1) Create a simple HDRI dome light
--   2) Generate Ray Tracing in One Weekend scene
--   3) Render to an image file

local function usage()
    print("Usage: tsdLua render_rtow_hdri.lua [options]")
    print("")
    print("Options:")
    print("  --out <prefix>     Output filename prefix (default: output)")
    print("  --library <name>   ANARI library name (default: visrtx)")
    print("  --width <int>      Image width in pixels (default: 1920)")
    print("  --height <int>     Image height in pixels (default: 1080)")
    print("  --samples <int>    Number of samples per pixel (default: 64)")
    print("  --frames <int>     Number of frames (default: 60)")
    print("  --help             Show this help")
end

local outPrefix = "output"
local libraryName = "visrtx"
local width = 1920
local height = 1080
local samples = 16
local numFrames = 1
local denoise = true

local i = 1
while arg and i <= #arg do
    local a = arg[i]
    if a == "--out" then
        i = i + 1
        outPrefix = arg[i] or outPrefix
    elseif a == "--library" then
        i = i + 1
        libraryName = arg[i] or libraryName
    elseif a == "--width" then
        i = i + 1
        width = tonumber(arg[i]) or width
    elseif a == "--height" then
        i = i + 1
        height = tonumber(arg[i]) or height
    elseif a == "--samples" then
        i = i + 1
        samples = tonumber(arg[i]) or samples
    elseif a == "--frames" then
        i = i + 1
        numFrames = tonumber(arg[i]) or numFrames
    elseif a == "--denoise" then
        denoise = true
    elseif a == "--help" or a == "-h" then
        usage()
        return
    else
        print("Unknown argument: " .. tostring(a))
        print("")
        usage()
        error("invalid argument")
    end
    i = i + 1
end

if width <= 0 or height <= 0 then
    error("width and height must be > 0")
end
if samples < 1 then
    error("samples must be >= 1")
end

print("RTOW + HDRI Render")
print("==================")
print("Library: " .. libraryName)
print("Output:  " .. outPrefix)
print(string.format("Size:    %dx%d", width, height))
print("Samples: " .. samples)
print("")

print("Creating HDRI dome light...")
tsd.io.generateHdriDome(scene)

local layer = scene:defaultLayer()
local rootXfm = scene:insertChildTransformNode(layer:root(), tsd.mat4.identity, "ROOT")

print("Generating RTOW scene...")
local rtowXfm = scene:insertChildTransformNode(rootXfm, tsd.mat4.identity, "RTOW")
tsd.io.generateRtow(scene, rtowXfm)

print("Remove the ground and center spheres")
local rtowRoot = rtowXfm:child(0)
local ground = rtowRoot:child(0)         -- "sphere_0"
local centerSphere = rtowRoot:child(484) -- "sphere_484"
scene:removeNode(ground)
scene:removeNode(centerSphere)

print("Generate Material ORB scene")
local moXfm = scene:insertChildTransformNode(rootXfm, tsd.scaling(8), "MaterialORB")
tsd.io.generateMaterialOrb(scene, moXfm)

print("Loading ANARI device...")
local device = tsd.render.loadDevice(libraryName)

print("Creating render index...")
local renderIndex = tsd.render.createRenderIndex(scene, device)
renderIndex:populate()

print("Computing scene bounds...")
local bounds = tsd.render.getWorldBounds(device, renderIndex)
local center = (bounds.min + bounds.max) * 0.5
local diag = bounds.max - bounds.min
local dist = tsd.length(diag) / 3.0
if dist < 1e-3 then
    dist = 5.0
end

local camera = tsd.CameraSetup.new()
camera.position = center + tsd.float3(0, 10, -dist)
camera.direction = tsd.normalize(center - camera.position)
camera.up = tsd.float3(0, 1, 0)
camera.fovy = 40.0
camera.aspect = width / height

print("Creating render pipeline...")
local pipeline = tsd.render.createPipeline(
    width, height, device, renderIndex, camera,
    { denoise = denoise }
)

-- Render frames, rotating 360 degrees around Y
print("Rendering " .. numFrames .. " frames...")
for frame = 0, numFrames - 1 do
    local angle = tsd.radians(frame * 360.0 / numFrames)
    local rot = tsd.rotation(tsd.float3(0, 1, 0), angle)
    rootXfm:setAsTransform(rot)
    scene:signalLayerChange(layer)

    local filename = string.format("%s_%04d.png", outPrefix, frame)
    tsd.render.renderToFile(pipeline, samples, filename, width, height)
    print(string.format("  frame %d/%d  angle=%6.1f°  -> %s",
        frame + 1, numFrames, math.deg(angle), filename))
end

print("")
print("Done: " .. outPrefix)
