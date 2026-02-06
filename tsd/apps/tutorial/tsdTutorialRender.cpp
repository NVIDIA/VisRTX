// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/core/scene/Scene.hpp>
// tsd_io
#include <tsd/io/procedural.hpp>
// tsd_rendering
#include <tsd/rendering/index/RenderIndexFlatRegistry.hpp>
// std
#include <cstdio>
// stb_image
#include "stb_image_write.h"

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

int main()
{
  // Create context //

  tsd::core::Scene scene;

  // Populate spheres //

  tsd::io::generate_randomSpheres(scene);

  auto light = scene.createObject<tsd::core::Light>("directional");
  light->setName("mainLight");
  light->setParameter("direction", float3(-1.f, 0.f, 0.f));
  light->setParameter("irradiance", 1.f);

  printf("%s\n", tsd::core::objectDBInfo(scene.objectDB()).c_str());

  // Setup ANARI device //

  anari::Library lib = anari::loadLibrary("environment", statusFunc);
  anari::Device device = anari::newDevice(lib, "default");

  // Setup render index //

  tsd::rendering::RenderIndexFlatRegistry rIdx(scene, device);
  rIdx.populate();

  // Create camera //

  auto camera = anari::newObject<anari::Camera>(device, "perspective");

  const float3 eye = {0.f, 0.f, -2.f};
  const float3 dir = {0.f, 0.f, 1.f};
  const float3 up = {0.f, 1.f, 0.f};

  anari::setParameter(device, camera, "position", eye);
  anari::setParameter(device, camera, "direction", dir);
  anari::setParameter(device, camera, "up", up);

  uint2 imageSize = {1200, 800};
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));

  anari::commitParameters(device, camera);

  // Create renderer //

  auto renderer = anari::newObject<anari::Renderer>(device, "default");
  const float4 backgroundColor = {0.1f, 0.1f, 0.1f, 1.f};
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
  stbi_write_png("output.png", fb.width, fb.height, 4, fb.data, 4 * fb.width);
  anari::unmap(device, frame, "channel.color");

  // Cleanup remaining ANARI objets //

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, frame);
  anari::release(device, device);
  anari::unloadLibrary(lib);

  return 0;
}
