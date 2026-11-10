// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Runtime callable-layout gate: all materials must sample a lit, front-facing
// surface through both renderer pipelines, not just compile matching
// signatures.
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using Vec3 = std::array<float, 3>;
using Vec4 = std::array<float, 4>;
using UVec2 = std::array<unsigned, 2>;

static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

static bool run(
    anari::Device d, const char *rendererType, const char *materialType)
{
  const std::array<Vec3, 4> positions = {
      Vec3{-2, -2, 0}, Vec3{2, -2, 0}, Vec3{2, 2, 0}, Vec3{-2, 2, 0}};
  auto geometry = anari::newObject<anari::Geometry>(d, "quad");
  anari::setParameterArray1D(
      d, geometry, "vertex.position", positions.data(), 4);
  anari::commitParameters(d, geometry);
  auto material = anari::newObject<anari::Material>(d, materialType);
  const std::string type(materialType);
  if (type == "mdl") {
    anari::setParameter(d, material, "sourceType", "code");
    anari::setParameter(d, material, "source", R"mdl(mdl 1.6;
import ::df::*;
export material sampling_diffuse() = material(
  surface: material_surface(scattering: df::diffuse_reflection_bsdf(
    tint: color(0.5))));
)mdl");
    anari::setParameter(d, material, "materialName", "sampling_diffuse");
  } else if (type == "matte")
    anari::setParameter(d, material, "color", Vec3{.5f, .5f, .5f});
  else {
    anari::setParameter(d, material, "baseColor", Vec3{.5f, .5f, .5f});
    anari::setParameter(d, material, "specular", 0.f);
    anari::setParameter(d, material, "metallic", 0.f);
  }
  anari::commitParameters(d, material);
  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geometry);
  anari::setAndReleaseParameter(d, surface, "material", material);
  anari::commitParameters(d, surface);

  const std::array<Vec3, 4> texels = {
      Vec3{1, 1, 1}, Vec3{1, 1, 1}, Vec3{1, 1, 1}, Vec3{1, 1, 1}};
  auto light = anari::newObject<anari::Light>(d, "hdri");
  anari::setAndReleaseParameter(
      d, light, "radiance", anari::newArray2D(d, texels.data(), 2, 2));
  anari::setParameter(d, light, "visible", false);
  anari::commitParameters(d, light);
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, world, "surface", &surface, 1);
  anari::setParameterArray1D(d, world, "light", &light, 1);
  anari::release(d, surface);
  anari::release(d, light);
  anari::commitParameters(d, world);
  auto camera = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, camera, "position", Vec3{0, 0, 1});
  anari::setParameter(d, camera, "direction", Vec3{0, 0, -1});
  anari::setParameter(d, camera, "up", Vec3{0, 1, 0});
  anari::setParameter(d, camera, "height", 2.f);
  anari::setParameter(d, camera, "aspect", 1.f);
  anari::commitParameters(d, camera);
  auto renderer = anari::newObject<anari::Renderer>(d, rendererType);
  anari::setParameter(d, renderer, "pixelSamples", 64);
  anari::setParameter(d, renderer, "ambientRadiance", 0.f);
  anari::setParameter(d, renderer, "denoise", false);
  anari::setParameter(d, renderer, "fireflyFilterMode", "none");
  anari::setParameter(d, renderer, "sampleLimit", 0);
  anari::commitParameters(d, renderer);
  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", UVec2{32, 32});
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(d, frame, "world", world);
  anari::setAndReleaseParameter(d, frame, "camera", camera);
  anari::setAndReleaseParameter(d, frame, "renderer", renderer);
  anari::commitParameters(d, frame);
  bool ok = true;
  // Uniform Lambertian rho=.5 under L=1 has outgoing radiance .5. A missing
  // continuation breaks the three-way MIS partition for PBR/raw MDL. Matte
  // intentionally returns a dead ray and has its own matching zero BSDF pdf.
  for (int launch = 0; launch < 2; ++launch) {
    anari::render(d, frame);
    anari::wait(d, frame);
    auto image = anari::map<Vec4>(d, frame, "channel.color");
    double mean = 0;
    if (!image.data || image.width != 32 || image.height != 32)
      ok = false;
    else {
      for (unsigned i = 0; i < 32 * 32; ++i) {
        const float value = image.data[i][0];
        ok &= std::isfinite(value);
        mean += value;
      }
      mean /= 32 * 32;
      ok &= std::abs(mean - .5) < .04;
    }
    anari::unmap(d, frame, "channel.color");
    std::printf("%s/%s launch %d mean %.9g: %s\n",
        rendererType,
        materialType,
        launch,
        mean,
        ok ? "ok" : "FAIL");
  }
  anari::release(d, frame);
  return ok;
}

int main()
{
  auto d = makeVisRTXDevice(status);
  bool ok = true;
  for (const char *renderer : {"quality", "interactive"}) {
    ok &= run(d, renderer, "matte");
    ok &= run(d, renderer, "physicallyBased");
#ifdef VISRTX_TEST_MDL_WRAPPER
    ok &= run(d, renderer, "physicallyBasedMDL");
    ok &= run(d, renderer, "mdl");
#endif
  }
  anari::release(d, d);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
