// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd
#include "render_pipeline/RenderPipeline.h"
#include "tsd/TSD.hpp"
// std
#include <cstdio>
// stb_image
#include "tsd_stb/stb_image_write.h"

using namespace tsd::literals;

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
#if 0
  else if (severity == ANARI_SEVERITY_INFO)
    fprintf(stderr, "[INFO ] %s\n", message);
  else if (severity == ANARI_SEVERITY_DEBUG)
    fprintf(stderr, "[DEBUG] %s\n", message);
#endif
}

int main()
{
  // Create context //

  tsd::Context ctx;

  // Populate spheres //

  tsd::generate_randomSpheres(ctx);

  auto light = ctx.createObject<tsd::Light>("directional");
  light->setName("mainLight");
  light->setParameter("direction"_t, tsd::float3(-1.f, 0.f, 0.f));
  light->setParameter("irradiance"_t, 1.f);

  printf("%s\n", tsd::objectDBInfo(ctx.objectDB()).c_str());

  // Setup ANARI device //

  printf("create ANARI device...");
  fflush(stdout);

  anari::Library lib = anari::loadLibrary("environment", statusFunc);
  anari::Device device = anari::newDevice(lib, "default");

  printf("done!\n");

  // Setup render index //

  printf("setup render index...");
  fflush(stdout);

  tsd::RenderIndexFlatRegistry rIdx(device);
  rIdx.populate(ctx);

  printf("done!\n");

  // Create camera //

  auto camera = anari::newObject<anari::Camera>(device, "perspective");

  const tsd::float3 eye = {0.f, 0.f, -2.f};
  const tsd::float3 dir = {0.f, 0.f, 1.f};
  const tsd::float3 up = {0.f, 1.f, 0.f};

  anari::setParameter(device, camera, "position", eye);
  anari::setParameter(device, camera, "direction", dir);
  anari::setParameter(device, camera, "up", up);

  tsd::uint2 imageSize = {1200, 800};
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));

  anari::commitParameters(device, camera);

  // Create renderer //

  auto renderer = anari::newObject<anari::Renderer>(device, "default");
  const tsd::float4 backgroundColor = {0.1f, 0.1f, 0.1f, 1.f};
  anari::setParameter(device, renderer, "background", backgroundColor);
  anari::setParameter(device, renderer, "ambientRadiance", 0.2f);
  anari::setParameter(device, renderer, "pixelSamples", 16);
  anari::setParameter(device, renderer, "denoise", true);
  anari::commitParameters(device, renderer);

  // Setup pipeline //

  printf("setup pipeline...");
  fflush(stdout);

  tsd::RenderPipeline rpipe(imageSize.x, imageSize.y);

  auto *arp = rpipe.emplace_back<tsd::AnariRenderPass>(device);
  arp->setWorld(rIdx.world());
  arp->setRenderer(renderer);
  arp->setCamera(camera);

  anari::release(device, camera);
  anari::release(device, renderer);

  auto *hrp = rpipe.emplace_back<tsd::OutlineRenderPass>();
  hrp->setOutlineId(0);

  printf("done!\n");

  // Render frame //

  printf("render frame...");
  fflush(stdout);

  rpipe.render();

  printf("done!\n");

  stbi_flip_vertically_on_write(1);
  stbi_write_png("pipeline.png",
      imageSize.x,
      imageSize.y,
      4,
      rpipe.getColorBuffer(),
      4 * imageSize.x);

  // Cleanup remaining ANARI objets //

  anari::release(device, device);
  anari::unloadLibrary(lib);

  return 0;
}
