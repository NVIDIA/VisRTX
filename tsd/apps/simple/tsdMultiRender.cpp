// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd
#include "tsd/TSD.hpp"
// std
#include <cstdio>
// stb_image
#include "tsd_stb/stb_image_write.h"

using float3 = anari::math::float3;
using float4 = anari::math::float4;
using uint2 = anari::math::uint2;

using namespace tsd::literals;

void statusFunc(const void *,
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

void renderSingleFrame(anari::Device d, anari::Frame f, const char *filename)
{
  anari::render(d, f);
  anari::wait(d, f);

  float duration = 0.f;
  anari::getProperty(d, f, "duration", duration, ANARI_NO_WAIT);

  printf("rendered frame '%s' in %fms\n", filename, duration * 1000);

  auto fb = anari::map<uint32_t>(d, f, "channel.color");
  stbi_flip_vertically_on_write(1);
  stbi_write_png(filename, fb.width, fb.height, 4, fb.data, 4 * fb.width);
  anari::unmap(d, f, "channel.color");
}

int main()
{
  // Create context //

  tsd::Context ctx;

  // Populate spheres //

  tsd::generate_randomSpheres(ctx, ctx.defaultLayer()->root(), true);

  // Setup ANARI device //

  anari::Library lib = anari::loadLibrary("environment", statusFunc);
  anari::Device device = anari::newDevice(lib, "default");

  // Setup render indexes //

  tsd::MultiUpdateDelegate ud;

  auto *rIdx1 = ud.emplace<tsd::RenderIndexFlatRegistry>(device);
  auto *rIdx2 = ud.emplace<tsd::RenderIndexFlatRegistry>(device);

  rIdx1->populate(ctx, false);
  rIdx2->populate(ctx, false);

  ctx.setUpdateDelegate(&ud);

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
  anari::setParameter(device, renderer, "ambientRadiance", 1.f);
  anari::commitParameters(device, renderer);

  // Create frame (top-level object) //

  auto frame = anari::newObject<anari::Frame>(device);

  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);

  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);

  // Render initial frames //

  anari::setParameter(device, frame, "world", rIdx1->world());
  anari::commitParameters(device, frame);
  renderSingleFrame(device, frame, "test1_1.png");

  anari::setParameter(device, frame, "world", rIdx2->world());
  anari::commitParameters(device, frame);
  renderSingleFrame(device, frame, "test2_1.png");

  // Mutate material
  ctx.getObject<tsd::Material>(0)->setParameter(
      "color"_t, float3(0.f, 1.f, 0.f));

  // Render updated frames //

  anari::setParameter(device, frame, "world", rIdx1->world());
  anari::commitParameters(device, frame);
  renderSingleFrame(device, frame, "test1_2.png");

  anari::setParameter(device, frame, "world", rIdx2->world());
  anari::commitParameters(device, frame);
  renderSingleFrame(device, frame, "test2_2.png");

  // Cleanup remaining ANARI objets //

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, frame);
  anari::release(device, device);
  anari::unloadLibrary(lib);

  return 0;
}
