-- Animated Monkey Grid Scene
-- Sequence:
--   1) Create a simple HDRI dome light
--   2) Generate a monkey mesh and instance it in a 10x10 grid
--   3) Create a camera orbit animation
--   4) Save the scene as a .tsd file

local function usage()
    print("Usage: tsdLua save_scene.lua [options]")
    print("")
    print("Options:")
    print("  --out <name>       Output filename without extension (default: output)")
    print("  --frames <int>     Number of keyframes for the orbit (default: 60)")
    print("  --help             Show this help")
end

local outPrefix = "output"
local numFrames = 60

local i = 1
while arg and i <= #arg do
    local a = arg[i]
    if a == "--out" then
        i = i + 1
        outPrefix = arg[i] or outPrefix
    elseif a == "--frames" then
        i = i + 1
        numFrames = tonumber(arg[i]) or numFrames
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

if numFrames < 1 then
    error("frames must be >= 1")
end

print("Animated Monkey Grid Scene")
print("==========================")
print("Output:  " .. outPrefix .. ".tsd")
print("Frames:  " .. numFrames)
print("")

-- 1) Build the scene -----------------------------------------------------

print("Creating HDRI dome light...")
tsd.io.generateHdriDome(scene)


local layer = scene:defaultLayer()
local rootXfm = scene:insertChildTransformNode(layer:root(), tsd.mat4.identity, "ROOT")

print("Generating monkey mesh...")
local monkeyXfm = scene:insertChildTransformNode(rootXfm, tsd.mat4.identity, "Monkey")
tsd.io.generateMonkey(scene, monkeyXfm)

-- 2) Instance with a transform array ------------------------------------

local gridRows = 10
local gridCols = 10
local spacing  = 3.0

print(string.format("Creating %dx%d instance grid...", gridCols, gridRows))

local xfmArray = scene:createArray("mat4", gridRows * gridCols)
local xfmData = {}

for row = 0, gridRows - 1 do
    for col = 0, gridCols - 1 do
        local x = (col - (gridCols - 1) / 2.0) * spacing
        local z = (row - (gridRows - 1) / 2.0) * spacing
        xfmData[row * gridCols + col + 1] = tsd.translation(tsd.float3(x, 0, z))
    end
end

xfmArray:setData(xfmData)
monkeyXfm:setAsTransformArray(xfmArray)

-- 3) Create a perspective camera ----------------------------------------

-- Hardcoded orbit parameters (sized for the 10x10 grid)
local gridExtent  = (math.max(gridRows, gridCols) - 1) * spacing
local orbitCenter = tsd.float3(0, 0, 0)
local orbitDist   = gridExtent * 0.8
local orbitHeight = gridExtent * 0.5
local fovy        = 40.0

local camera      = scene:createCamera("perspective", { fovy = fovy / 2.0 })
camera.name       = "turntable"
camera:setParameter("up", tsd.float3(0, 1, 0))
camera:setParameter("aspect", 16.0 / 9.0)

-- 4) Compute orbit keyframes --------------------------------------------

print("Computing " .. numFrames .. " orbit keyframes...")

local posArray = scene:createArray("float3", numFrames)
local dirArray = scene:createArray("float3", numFrames)

local posData = {}
local dirData = {}

for frame = 0, numFrames - 1 do
    local angle = tsd.radians(frame * 360.0 / numFrames)
    local sx = math.sin(angle)
    local cz = math.cos(angle)

    local pos = orbitCenter + tsd.float3(orbitDist * sx, orbitHeight, orbitDist * cz)
    local dir = tsd.normalize(orbitCenter - pos)

    posData[frame + 1] = pos
    dirData[frame + 1] = dir
end

posArray:setData(posData)
dirArray:setData(dirData)

-- 5) Create animation ----------------------------------------------------

print("Creating turntable animation...")

local animation = scene:addAnimation("turntable")
animation:setAsTimeSteps(camera, { "position", "direction" }, { posArray, dirArray })
scene:setAnimationIncrement(1.0 / numFrames)

-- 6) Save with viewer state ----------------------------------------------

local camDistance = math.sqrt(orbitHeight * orbitHeight + orbitDist * orbitDist)
local camElevation = math.deg(math.atan(orbitHeight, orbitDist))

local filename = outPrefix .. ".tsd"
print("Saving scene to " .. filename .. "...")
-- State table mirrors the DataTree layout expected by tsdViewer
tsd.io.saveScene(scene, filename, {
    windows = {
        Viewport = {
            anariLibrary = "visrtx",
            camera = {
                at = orbitCenter,
                distance = camDistance,
                azel = tsd.float2(180.0, camElevation),
                up = 1,
            },
        },
    },
})

print("")
print("Done: " .. filename)
