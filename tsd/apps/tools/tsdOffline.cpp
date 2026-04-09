// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <tsd/app/Context.h>
#include <tsd/rendering/pipeline/ImagePipeline.h>
#include <tsd/core/Logging.hpp>
#include <tsd/core/Timer.hpp>
#include <tsd/scene/Scene.hpp>
#include <tsd/io/procedural.hpp>
#include <tsd/io/serialization.hpp>
#include <tsd/rendering/index/RenderIndexAllLayers.hpp>
#include <tsd/rendering/view/ManipulatorToAnari.hpp>
#include "stb_image_write.h"

#ifdef TSD_USE_MPI
#include <mpi.h>
#endif

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

static std::unique_ptr<tsd::rendering::RenderIndexAllLayers> g_renderIndex;
static std::unique_ptr<tsd::rendering::ImagePipeline> g_renderPipeline;
static tsd::core::Timer g_timer;
static tsd::rendering::Manipulator g_manipulator;
static std::vector<tsd::rendering::CameraPose> g_cameraPoses;
static std::unique_ptr<tsd::app::Context> g_ctx;

static tsd::core::Token g_deviceName;
static anari::Library g_library{nullptr};
static anari::Device g_device{nullptr};
static anari::Camera g_camera{nullptr};

struct Config
{
  tsd::math::float3 cameraPos = {0.f, 0.f, 3.f};
  tsd::math::float3 cameraLookAt = {0.f, 0.f, 0.f};
  tsd::math::float3 cameraUp = {0.f, 1.f, 0.f};
  float fovy = 40.f;
  bool autoCamera = true;

  std::string rendererName = "default";
  std::string outputFile = "tsdOffline.png";

  std::string animOutputDir;
  std::string animPrefix = "frame_";

  tsd::math::float4 background = {0.05f, 0.05f, 0.05f, 1.f};
  float ambientRadiance = 0.25f;
  tsd::math::float3 ambientColor = {1.f, 1.f, 1.f};
  bool hasBackground = false;

  struct DirectionalLight
  {
    tsd::math::float3 direction;
    tsd::math::float3 color;
    float irradiance;
  };
  std::vector<DirectionalLight> directionalLights;
};

static Config g_config;

static void printUsage(const char *programName)
{
  std::cout << "Usage: " << programName << " [options]\n\n";
  std::cout << "Rendering Options:\n";
  std::cout << "  -w, --width <int>          Frame width (default: 1024)\n";
  std::cout << "  -h, --height <int>         Frame height (default: 768)\n";
  std::cout
      << "  -s, --samples <int>        Samples per pixel (default: 128)\n";
  std::cout
      << "  -o, --output <file>        Output filename (default: tsdOffline.png)\n";
  std::cout
      << "  --lib <name>               ANARI library name (default: TSD_ANARI_LIBRARIES[0], environment, or visrtx)\n";
  std::cout
      << "  --renderer <name>          Renderer name (default: default)\n";
  std::cout << "  --campos <x y z>           Camera position (3 floats)\n";
  std::cout << "  --lookpos <x y z>          Camera look-at point (3 floats)\n";
  std::cout << "  --upvec <x y z>            Camera up vector (3 floats)\n";
  std::cout
      << "  --fovy <float>             Field of view Y in degrees (default: 40)\n";
  std::cout << "  --aperture <float>         Aperture radius (default: 0)\n";
  std::cout << "  --focus <float>            Focus distance (default: 1)\n";
  std::cout << "\n";
  std::cout << "Lighting and Background:\n";
  std::cout
      << "  --bg-color <r g b a>       Background color (4 floats, default: 0.05 0.05 0.05 1.0)\n";
  std::cout
      << "  --no-bg                    Set background to black (0 0 0 0)\n";
  std::cout
      << "  --ambient <float>          Ambient radiance (default: 0.25)\n";
  std::cout
      << "  --ambient-color <r g b>    Ambient color (3 floats, default: 1.0 1.0 1.0)\n";
  std::cout << "  --dir-light <dx dy dz> <r g b> <intensity>\n";
  std::cout
      << "                             Add directional light (direction + color + intensity)\n";
  std::cout << "  --help                     Show this help message\n";
  std::cout << "\n";
  std::cout << "Animation Options:\n";
  std::cout
      << "  --anim-out-dir <dir>       Output directory for animation frames\n";
  std::cout
      << "                             (enables animation mode; frames saved as\n";
  std::cout
      << "                              <dir>/<prefix><NNNN>.png)\n";
  std::cout
      << "  --anim-prefix <prefix>     Filename prefix for animation frames\n";
  std::cout
      << "                             (default: frame_)\n";
  std::cout
      << "  --num-frames <int>         Number of frames to render when scene has\n";
  std::cout
      << "                             no animation data (default: 1)\n";
#ifdef TSD_USE_MPI
  std::cout << "\n";
  std::cout
      << "MPI: when run with mpirun/srun, each rank renders an interleaved subset\n";
  std::cout
      << "     of animation frames (rank k renders frames k, k+N, k+2N, ...).\n";
#endif
  std::cout << "\n";
  std::cout << "Importer Options:\n";
  std::cout << "  -tsd <file>                Load TSD scene file\n";
  std::cout << "  -gltf <file...>            Import GLTF/GLB files\n";
  std::cout << "  -obj <file...>             Import Wavefront OBJ files\n";
  std::cout << "  -ply <file...>             Import PLY files\n";
  std::cout
      << "  -volume <file...>          Import volume data (RAW, NVDB, VTI, etc.)\n";
  std::cout << "  -hdri <file>               Set HDRI environment map\n";
  std::cout << "  -silo <file...>            Import Silo files\n";
  std::cout << "  -usd <file...>             Import USD files\n";
  std::cout
      << "  -l, --layer <name>         Specify layer name for following imports\n";
  std::cout
      << "  -assimp <file...>          Import via ASSIMP (supports many formats)\n";
  std::cout << "  -axyz <file...>            Import AXYZ point cloud files\n";
  std::cout << "  -e57xyz <file...>          Import E57 point cloud files\n";
  std::cout << "  -pdb <file...>             Import PDB molecule files\n";
  std::cout << "  -swc <file...>             Import SWC neuron files\n";
  std::cout << "  -trk <file...>             Import TRK track files\n";
  std::cout << "  -nbody <file...>           Import N-body simulation files\n";
  std::cout << "\n";
  std::cout << "Examples:\n";
  std::cout << "  # Render a GLTF model\n";
  std::cout << "  " << programName << " -gltf model.glb -o output.png\n";
  std::cout << "\n";
  std::cout << "  # Render volume data at custom resolution\n";
  std::cout << "  " << programName
            << " -volume density.raw -w 2048 -h 2048 -s 256 -o volume.png\n";
  std::cout << "\n";
  std::cout << "  # Combine multiple formats\n";
  std::cout << "  " << programName
            << " -obj mesh.obj -hdri env.hdr -w 1920 -h 1080 -o render.png\n";
  std::cout << "\n";
  std::cout << "  # With custom camera\n";
  std::cout
      << "  " << programName
      << " -gltf scene.glb --campos 10 10 10 --lookpos 0 0 0 -o out.png\n";
  std::cout << "\n";
  std::cout
      << "If no importer flags are specified, a default empty scene will be created.\n";
  std::cout
      << "If camera is not specified, it will be computed from scene bounds.\n";
}

static tsd::math::float3 parseFloat3(const char **argv, int &i)
{
  tsd::math::float3 result;
  result.x = std::stof(argv[++i]);
  result.y = std::stof(argv[++i]);
  result.z = std::stof(argv[++i]);
  return result;
}

// Parse rendering-specific options and build a new argv with importer options
// Returns the new argc for importer parsing, or -1 on error
static int parseRenderingOptions(
    int argc, const char *argv[], std::vector<const char *> &importerArgv)
{
  // First element is always the program name
  importerArgv.push_back(argv[0]);

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--help") {
      printUsage(argv[0]);
      return -1;
    } else if (arg == "-w" || arg == "--width") {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << arg << " requires an argument\n";
        return -1;
      }
      g_ctx->offline.frame.width = std::stoi(argv[++i]);
    } else if (arg == "-h" || arg == "--height") {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << arg << " requires an argument\n";
        return -1;
      }
      g_ctx->offline.frame.height = std::stoi(argv[++i]);
    } else if (arg == "-s" || arg == "--samples") {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << arg << " requires an argument\n";
        return -1;
      }
      g_ctx->offline.frame.samples = std::stoi(argv[++i]);
    } else if (arg == "-o" || arg == "--output") {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << arg << " requires an argument\n";
        return -1;
      }
      g_config.outputFile = argv[++i];
    } else if (arg == "--lib") {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << arg << " requires an argument\n";
        return -1;
      }
      g_ctx->offline.renderer.libraryName = argv[++i];
    } else if (arg == "--renderer") {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << arg << " requires an argument\n";
        return -1;
      }
      g_config.rendererName = argv[++i];
    } else if (arg == "--campos") {
      if (i + 3 >= argc) {
        std::cerr << "Error: --campos requires 3 arguments (x y z)\n";
        return -1;
      }
      g_config.cameraPos = parseFloat3(argv, i);
      g_config.autoCamera = false;
    } else if (arg == "--lookpos") {
      if (i + 3 >= argc) {
        std::cerr << "Error: --lookpos requires 3 arguments (x y z)\n";
        return -1;
      }
      g_config.cameraLookAt = parseFloat3(argv, i);
      g_config.autoCamera = false;
    } else if (arg == "--upvec") {
      if (i + 3 >= argc) {
        std::cerr << "Error: --upvec requires 3 arguments (x y z)\n";
        return -1;
      }
      g_config.cameraUp = parseFloat3(argv, i);
    } else if (arg == "--fovy") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --fovy requires an argument\n";
        return -1;
      }
      g_config.fovy = std::stof(argv[++i]);
    } else if (arg == "--aperture") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --aperture requires an argument\n";
        return -1;
      }
      g_ctx->offline.camera.apertureRadius = std::stof(argv[++i]);
    } else if (arg == "--focus") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --focus requires an argument\n";
        return -1;
      }
      g_ctx->offline.camera.focusDistance = std::stof(argv[++i]);
    } else if (arg == "--bg-color") {
      if (i + 4 >= argc) {
        std::cerr << "Error: --bg-color requires 4 arguments (r g b a)\n";
        return -1;
      }
      g_config.background.x = std::stof(argv[++i]);
      g_config.background.y = std::stof(argv[++i]);
      g_config.background.z = std::stof(argv[++i]);
      g_config.background.w = std::stof(argv[++i]);
      g_config.hasBackground = true;
    } else if (arg == "--no-bg") {
      g_config.background = {0.f, 0.f, 0.f, 0.f};
      g_config.hasBackground = true;
    } else if (arg == "--ambient") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --ambient requires an argument\n";
        return -1;
      }
      g_config.ambientRadiance = std::stof(argv[++i]);
    } else if (arg == "--ambient-color") {
      if (i + 3 >= argc) {
        std::cerr << "Error: --ambient-color requires 3 arguments (r g b)\n";
        return -1;
      }
      g_config.ambientColor.x = std::stof(argv[++i]);
      g_config.ambientColor.y = std::stof(argv[++i]);
      g_config.ambientColor.z = std::stof(argv[++i]);
    } else if (arg == "--anim-out-dir") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --anim-out-dir requires an argument\n";
        return -1;
      }
      g_config.animOutputDir = argv[++i];
    } else if (arg == "--anim-prefix") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --anim-prefix requires an argument\n";
        return -1;
      }
      g_config.animPrefix = argv[++i];
    } else if (arg == "--num-frames") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --num-frames requires an argument\n";
        return -1;
      }
      g_ctx->offline.frame.numFrames = std::stoi(argv[++i]);
    } else if (arg == "--dir-light") {
      if (i + 7 >= argc) {
        std::cerr
            << "Error: --dir-light requires 7 arguments (dx dy dz r g b intensity)\n";
        return -1;
      }
      Config::DirectionalLight light;
      light.direction.x = std::stof(argv[++i]);
      light.direction.y = std::stof(argv[++i]);
      light.direction.z = std::stof(argv[++i]);
      light.color.x = std::stof(argv[++i]);
      light.color.y = std::stof(argv[++i]);
      light.color.z = std::stof(argv[++i]);
      light.irradiance = std::stof(argv[++i]);
      g_config.directionalLights.push_back(light);
    } else {
      // Not a rendering option, pass it to importer parsing
      importerArgv.push_back(argv[i]);
    }
  }

  return static_cast<int>(importerArgv.size());
}

static void loadANARIDevice()
{
  auto statusFunc = [](const void *,
                        ANARIDevice,
                        ANARIObject,
                        ANARIDataType,
                        ANARIStatusSeverity severity,
                        ANARIStatusCode,
                        const char *message) {
    if (severity == ANARI_SEVERITY_FATAL_ERROR) {
      fprintf(stderr, "[ANARI][FATAL] %s\n", message);
      std::exit(1);
    } else if (severity == ANARI_SEVERITY_ERROR)
      fprintf(stderr, "[ANARI][ERROR] %s\n", message);
  };

  const auto &libraryName = g_ctx->offline.renderer.libraryName;
  g_deviceName = libraryName;

  printf("Loading ANARI device from '%s' library...", libraryName.c_str());
  fflush(stdout);

  g_timer.start();
  g_library = anari::loadLibrary(libraryName.c_str(), statusFunc);
  g_device = anari::newDevice(g_library, "default");
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void initTSDRenderIndex()
{
  printf("Initializing TSD render index...");
  fflush(stdout);

  g_timer.start();
  g_renderIndex = std::make_unique<tsd::rendering::RenderIndexAllLayers>(
      g_ctx->tsd.scene, g_deviceName, g_device);
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void populateTSDScene()
{
  printf("Importing scene data...");
  fflush(stdout);

  g_timer.start();

  g_ctx->setupSceneFromCommandLine();

  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void populateRenderIndex()
{
  printf("Populating TSD render index...");
  fflush(stdout);

  g_timer.start();
  g_renderIndex->populate();
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void setupLights()
{
  printf("Setting up lights...");
  fflush(stdout);

  g_timer.start();

  // Add user-specified directional lights
  for (const auto &lightConfig : g_config.directionalLights) {
    auto light =
        g_ctx->tsd.scene.createObject<tsd::scene::Light>("directional");
    light->setParameter("direction", lightConfig.direction);
    light->setParameter("color", lightConfig.color);
    light->setParameter("irradiance", lightConfig.irradiance);
  }

  // If no lights were specified and scene has none, add viewer-default lights
  if (g_config.directionalLights.empty()
      && g_ctx->tsd.scene.numberOfObjects(ANARI_LIGHT) == 0) {
    tsd::io::generate_default_lights(g_ctx->tsd.scene);
    printf("(added default light)...");
    fflush(stdout);
  } else {
    printf("(%zu light(s) in scene)...",
        g_ctx->tsd.scene.numberOfObjects(ANARI_LIGHT));
    fflush(stdout);
  }

  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void setupCameraManipulator()
{
  printf("Setting up camera...");
  fflush(stdout);

  g_timer.start();

  if (g_cameraPoses.empty()) {
    tsd::rendering::CameraPose pose;

    if (g_config.autoCamera) {
      printf("from world bounds...");
      fflush(stdout);
      pose = g_renderIndex->computeDefaultView();
    } else {
      printf("from command line...");
      fflush(stdout);

      pose.lookat = g_config.cameraLookAt;
      pose.fixedDist =
          tsd::math::length(g_config.cameraPos - g_config.cameraLookAt);

      auto dir =
          tsd::math::normalize(g_config.cameraPos - g_config.cameraLookAt);
      float azimuth = std::atan2(dir.x, dir.z) * 180.f / M_PI;
      float elevation = std::asin(dir.y) * 180.f / M_PI;
      pose.azeldist = {azimuth, elevation, pose.fixedDist};
      pose.upAxis = static_cast<int>(tsd::rendering::UpAxis::POS_Y);
    }

    g_cameraPoses.push_back(std::move(pose));
  }

  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void setupImagePipeline()
{
  const auto frameWidth = g_ctx->offline.frame.width;
  const auto frameHeight = g_ctx->offline.frame.height;

  printf("Setting up render pipeline (%u x %u)...", frameWidth, frameHeight);
  fflush(stdout);

  g_timer.start();
  g_renderPipeline =
      std::make_unique<tsd::rendering::ImagePipeline>(frameWidth, frameHeight);

  g_camera = anari::newObject<anari::Camera>(g_device, "perspective");
  anari::setParameter(
      g_device, g_camera, "aspect", frameWidth / float(frameHeight));
  anari::setParameter(
      g_device, g_camera, "fovy", anari::radians(g_config.fovy));
  anari::setParameter(g_device,
      g_camera,
      "apertureRadius",
      g_ctx->offline.camera.apertureRadius);
  anari::setParameter(g_device,
      g_camera,
      "focusDistance",
      g_ctx->offline.camera.focusDistance);
  anari::commitParameters(g_device, g_camera);

  auto r = anari::newObject<anari::Renderer>(
      g_device, g_config.rendererName.c_str());

  // Set background and ambient lighting
  anari::setParameter(g_device, r, "background", g_config.background);
  anari::setParameter(g_device, r, "ambientRadiance", g_config.ambientRadiance);
  anari::setParameter(g_device, r, "ambientColor", g_config.ambientColor);

  anari::commitParameters(g_device, r);

  auto *arp =
      g_renderPipeline->emplace_back<tsd::rendering::AnariSceneRenderPass>(
          g_device);
  arp->setWorld(g_renderIndex->world());
  arp->setRenderer(r);
  arp->setCamera(g_camera);
  arp->setRunAsync(false);

  anari::release(g_device, r);

  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void renderFrames()
{
#ifdef TSD_USE_MPI
  int mpiRank = 0, mpiSize = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
#else
  const int mpiRank = 0, mpiSize = 1;
#endif

  const bool animMode = !g_config.animOutputDir.empty();

  int numFrames = 1;
  if (animMode) {
    auto &animMgr = g_ctx->tsd.animationMgr;
    numFrames = !animMgr.animations().empty()
        ? animMgr.getAnimationTotalFrames()
        : g_ctx->offline.frame.numFrames;
  }

  const auto frameSamples = g_ctx->offline.frame.samples;
  const auto frameWidth = g_ctx->offline.frame.width;
  const auto frameHeight = g_ctx->offline.frame.height;

  if (mpiRank == 0) {
    if (animMode)
      printf("Rendering %d frame(s) (%u spp)...\n", numFrames, frameSamples);
    else
      printf("Rendering frame (%u spp)...\n", frameSamples);
    fflush(stdout);
  }

  stbi_flip_vertically_on_write(1);

  const auto &pose = g_cameraPoses[0];
  g_manipulator.setConfig(pose);
  tsd::rendering::updateCameraParametersPerspective(
      g_device, g_camera, g_manipulator);
  anari::commitParameters(g_device, g_camera);

  for (int frameIndex = 0; frameIndex < numFrames; ++frameIndex) {
    if (mpiSize > 1 && frameIndex % mpiSize != mpiRank)
      continue;

    if (animMode)
      g_ctx->tsd.animationMgr.setAnimationFrame(frameIndex);

    g_timer.start();
    for (int s = 0; s < (int)frameSamples; ++s) {
      g_renderPipeline->render();
      if ((s + 1) % 10 == 0 || s == (int)frameSamples - 1) {
        if (animMode)
          printf("[rank %d] frame %d/%d: %d/%u spp\r",
              mpiRank,
              frameIndex + 1,
              numFrames,
              s + 1,
              frameSamples);
        else
          printf("...rendered %d/%u samples\r", s + 1, frameSamples);
        fflush(stdout);
      }
    }
    printf("\n");
    g_timer.end();

    std::string outPath;
    if (animMode) {
      std::ostringstream ss;
      ss << g_config.animOutputDir << "/" << g_config.animPrefix
         << std::setfill('0') << std::setw(4) << frameIndex << ".png";
      outPath = ss.str();
    } else {
      outPath = g_config.outputFile;
    }

    stbi_write_png(outPath.c_str(),
        frameWidth,
        frameHeight,
        4,
        g_renderPipeline->getColorBuffer(),
        4 * frameWidth);

    printf("[rank %d] written: %s (%.2f ms)\n",
        mpiRank,
        outPath.c_str(),
        g_timer.milliseconds());
  }
}

static void cleanup()
{
  printf("Cleanup objects...");
  fflush(stdout);

  g_timer.start();
  g_renderPipeline.reset();
  g_renderIndex.reset();
  anari::release(g_device, g_camera);
  anari::release(g_device, g_device);
  anari::unloadLibrary(g_library);
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

int main(int argc, const char *argv[])
{
#ifdef TSD_USE_MPI
  MPI_Init(&argc, (char ***)&argv);
  int mpiRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
#else
  const int mpiRank = 0;
#endif

  // Enable TSD logging to stdout so we can see import errors
  tsd::core::setLogToStdout();

  // Initialize Context first so it can be used for parsing
  g_ctx = std::make_unique<tsd::app::Context>();
  // Default library: TSD_ANARI_LIBRARIES[0], then "environment" (reads
  // ANARI_LIBRARY), then "visrtx"
  std::string defaultLibrary = "visrtx";
  if (getenv("ANARI_LIBRARY"))
    defaultLibrary = "environment";
  if (const char *tsdLibs = getenv("TSD_ANARI_LIBRARIES")) {
    std::string envStr = tsdLibs;
    auto comma = envStr.find(',');
    std::string first =
        (comma != std::string::npos) ? envStr.substr(0, comma) : envStr;
    first.erase(0, first.find_first_not_of(" \t"));
    first.erase(first.find_last_not_of(" \t") + 1);
    if (!first.empty())
      defaultLibrary = first;
  }
  g_ctx->offline.renderer.libraryName = defaultLibrary; // overridden by --lib

  // Two-pass command line parsing:
  // 1. Extract rendering options (width, height, samples, etc.)
  // 2. Pass remaining args (importer flags and filenames) to Context
  std::vector<const char *> importerArgv;
  int importerArgc = parseRenderingOptions(argc, argv, importerArgv);
  if (importerArgc < 0) {
    return 1;
  }

  // Let Context parse importer options (-gltf, -obj, -volume, etc.)
  g_ctx->parseCommandLine(importerArgc, importerArgv.data());

  if (mpiRank == 0) {
    printf("tsdOffline - Headless TSD Renderer\n");
    printf("===================================\n");
    printf("Resolution: %ux%u\n",
        g_ctx->offline.frame.width,
        g_ctx->offline.frame.height);
    printf("Samples: %u\n", g_ctx->offline.frame.samples);
    printf("Library: %s\n", g_ctx->offline.renderer.libraryName.c_str());
    printf("Renderer: %s\n", g_config.rendererName.c_str());
    if (g_config.animOutputDir.empty())
      printf("Output: %s\n", g_config.outputFile.c_str());
    else
      printf("Animation output: %s/%s<NNNN>.png\n",
          g_config.animOutputDir.c_str(),
          g_config.animPrefix.c_str());
    printf("\n");
  }

  // Context already initializes its scene, no separate initialization needed
  loadANARIDevice();
  populateTSDScene(); // Import data into scene FIRST
  setupLights(); // Then add lights
  initTSDRenderIndex(); // THEN create render index with populated scene
  populateRenderIndex();
  setupCameraManipulator();
  setupImagePipeline();
  renderFrames();
  cleanup();

  g_ctx.reset();

#ifdef TSD_USE_MPI
  MPI_Finalize();
#endif

  return 0;
}
