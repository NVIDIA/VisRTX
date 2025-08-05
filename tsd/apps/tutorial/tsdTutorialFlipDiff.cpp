// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/core/scene/Context.hpp>
// tsd_io
#include <tsd/io/procedural.hpp>
// tsd_rendering
#include <tsd/rendering/index/RenderIndexFlatRegistry.hpp>
// stb_image
#include "stb_image_write.h"
// FLIP
#include <FLIP.h>
// std
#include <chrono>
#include <cstdio>
#include <vector>

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
#if 0
  else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING)
    fprintf(stderr, "[PERF ] %s\n", message);
  else if (severity == ANARI_SEVERITY_INFO)
    fprintf(stderr, "[INFO ] %s\n", message);
  else if (severity == ANARI_SEVERITY_DEBUG)
    fprintf(stderr, "[DEBUG] %s\n", message);
#endif
}

static std::vector<float3> extractColors(anari::Device d, anari::Frame f)
{
  std::vector<float3> out;

  auto fb = anari::map<float4>(d, f, "channel.color");

  const size_t nPixels = fb.width * fb.height;
  if (fb.pixelType == ANARI_FLOAT32_VEC4 && nPixels > 0) {
    out.resize(fb.width * fb.height);
    std::transform(fb.data, fb.data + nPixels, out.data(), [](const float4 &c) {
      return float3(c.x, c.y, c.z);
    });
  } else {
    printf("ERROR: Frame did not map as 'ANARI_FLOAT32_VEC4' or was empty\n");
  }

  anari::unmap(d, f, "channel.color");

  return out;
}

int main()
{
  // Create context //

  tsd::core::Context ctx;

  // Populate spheres //

  tsd::io::generate_randomSpheres(ctx);

  auto light =
      ctx.createObject<tsd::core::Light>(tsd::core::tokens::light::directional);
  light->setName("mainLight");
  light->setParameter("direction", float3(-1.f, 0.f, 0.f));
  light->setParameter("irradiance", 1.f);

  printf("%s\n", tsd::core::objectDBInfo(ctx.objectDB()).c_str());

  // Setup ANARI device //

  anari::Library lib = anari::loadLibrary("visrtx", statusFunc);
  anari::Device device = anari::newDevice(lib, "default");

  // Setup render index //

  tsd::rendering::RenderIndexFlatRegistry rIdx(ctx, device);
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
  anari::setParameter(device, renderer, "ambientRadiance", 0.5f);
  anari::setParameter(device, renderer, "pixelSamples", 1);
  anari::setParameter(device, renderer, "denoise", false);
  anari::commitParameters(device, renderer);

  // Create frame (top-level object) //

  auto frame = anari::newObject<anari::Frame>(device);

  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);

  anari::setParameter(device, frame, "world", rIdx.world());
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);

  anari::commitParameters(device, frame);

  // Render noisy frame //

  anari::render(device, frame);
  anari::wait(device, frame);

  float duration = 0.f;
  anari::getProperty(device, frame, "duration", duration, ANARI_NO_WAIT);

  printf("rendered noisy frame in %fms\n", duration * 1000);

  auto testImage = extractColors(device, frame);

  // Render clean image //

  anari::setParameter(device, renderer, "pixelSamples", 64);
  anari::setParameter(device, renderer, "denoise", true);
  anari::commitParameters(device, renderer);

  anari::render(device, frame);
  anari::wait(device, frame);

  anari::getProperty(device, frame, "duration", duration, ANARI_NO_WAIT);

  printf("rendered noisy frame in %fms\n", duration * 1000);

  auto referenceImage = extractColors(device, frame);

  // Do FLIP diff //

  std::vector<float> flipDiff(referenceImage.size());

  FLIP::Parameters fparams;
  auto start = std::chrono::steady_clock::now();
  FLIP::computeFLIP(false,
      fparams,
      imageSize.x,
      imageSize.y,
      (const float *)referenceImage.data(),
      (const float *)testImage.data(),
      flipDiff.data());
  auto end = std::chrono::steady_clock::now();
  duration = std::chrono::duration<float>(end - start).count();

  stbi_write_hdr("flip.hdr", imageSize.x, imageSize.y, 1, flipDiff.data());

  printf("computed FLIP diff in %fms\n", duration * 1000);

  // Cleanup remaining ANARI objets //

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, frame);
  anari::release(device, device);
  anari::unloadLibrary(lib);

  return 0;
}
