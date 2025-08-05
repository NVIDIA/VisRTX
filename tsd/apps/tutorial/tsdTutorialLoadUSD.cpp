// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/core/scene/Context.hpp>
// tsd_io
#include <tsd/io/importers.hpp>
// tsd_rendering
#include <tsd/rendering/index/RenderIndexFlatRegistry.hpp>
// std
#include <cstdio>
// stb_image
#include "stb_image_write.h"

static std::string g_libraryName = "environment";
static std::string g_filename;

using float3 = tsd::math::float3;
using float4 = tsd::math::float4;
using uint2 = tsd::math::uint2;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR)
    fprintf(stderr, "[FATAL] %s\n", message);
  else if (severity == ANARI_SEVERITY_ERROR)
    fprintf(stderr, "[ERROR] %s\n", message);
  else if (severity == ANARI_SEVERITY_WARNING)
    fprintf(stderr, "[WARN ] %s\n", message);
  else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING)
    fprintf(stderr, "[PERF ] %s\n", message);
  else if (severity == ANARI_SEVERITY_INFO)
    fprintf(stderr, "[INFO ] %s\n", message);
  else if (severity == ANARI_SEVERITY_DEBUG)
    fprintf(stderr, "[DEBUG] %s\n", message);
}

static void printUsage()
{
  std::cout << "./tsd_load_usd [{--help|-h}] file.usd\n";
  std::exit(0);
}

static void parseCommandLine(int argc, char *argv[])
{
  if (argc < 2)
    printUsage();

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
      printUsage();
    else if (arg == "-l" || arg == "--library")
      g_libraryName = argv[++i];
    else
      g_filename = arg;
  }
}

int main(int argc, char *argv[])
{
  parseCommandLine(argc, argv);

  // Create context //

  tsd::core::Context ctx;

  // Populate spheres //

  tsd::io::import_USD(ctx, g_filename.c_str());

  // Setup ANARI device //

  anari::Library lib = anari::loadLibrary("helide", statusFunc);
  anari::Device device = anari::newDevice(lib, "default");

  // Setup render index //

  tsd::rendering::RenderIndexFlatRegistry rIdx(ctx, device);
  rIdx.populate();

  // Create camera //

  auto camera = anari::newObject<anari::Camera>(device, "perspective");

  // Kitchen scene
  // const float3 eye = {75.0f, -600.0, 150.0f};
  // McDo scene
  const float3 eye = {0.5f, 0.0f, 10.0f};

  const float3 dir = {0.0f, 0.0f, -1.0f};
  const float3 up = {0.0f, 0.0f, 1.0f};

  anari::setParameter(device, camera, "position", eye);
  anari::setParameter(device, camera, "direction", dir);
  anari::setParameter(device, camera, "up", up);

  uint2 imageSize = {1200, 800};
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));

  anari::commitParameters(device, camera);

  // Create renderer //

  auto renderer = anari::newObject<anari::Renderer>(device, "default");
  const float4 backgroundColor = {1.0f, 1.0f, 1.0f, 0.2f};
  anari::setParameter(device, renderer, "background", backgroundColor);
  anari::setParameter(device, renderer, "ambientRadiance", 0.2f);
  anari::setParameter(device, renderer, "pixelSamples", 16);
  anari::setParameter(device, renderer, "denoise", true);
  anari::commitParameters(device, renderer);

  // Create frame (top-level object) //

  auto frame = anari::newObject<anari::Frame>(device);

  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);

  anari::setParameter(device, frame, "world", rIdx.world());
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);

  anari::commitParameters(device, frame);

  // Render frame and print out duration property //

  anari::render(device, frame);
  anari::wait(device, frame);

  float duration = 0.f;
  anari::getProperty(device, frame, "duration", duration, ANARI_NO_WAIT);

  printf("rendered frame in %fms\n", duration * 1000);

  stbi_flip_vertically_on_write(1);
  auto fb = anari::map<uint32_t>(device, frame, "channel.color");
  stbi_write_png("load_usd.png", fb.width, fb.height, 4, fb.data, 4 * fb.width);
  anari::unmap(device, frame, "channel.color");

  // Cleanup remaining ANARI objets //

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, frame);
  anari::release(device, device);
  anari::unloadLibrary(lib);

  return 0;
}
