/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Quality combines folded env-CDF NEE, unconditional cosine NEE, and BSDF
 * escape. Only the CDF density carries the per-instance Light Pick mass.
 * Check linearity with a directional light and analytic Lambertian energy
 * rho*L for uniform, nonuniform, multiple, hidden, and one-texel HDRIs.
 * Matte isolates the NEE partition; diffuse PBR also exercises escape MIS.
 * Hidden lights must illuminate without becoming visible camera backgrounds.
 * Measurements use a linear float buffer with the firefly filter disabled.
 */

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject source,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR
      || severity == ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
    std::exit(1);
  }
}

static constexpr uvec2 IMAGE_SIZE = {256, 256};
static constexpr int PIXEL_SAMPLES = 256;

struct HdriMap
{
  float upperRadiance{0.6f};
  float lowerRadiance{0.6f};
  bool visible{true};
  uvec2 size{64, 128};
};

static anari::Light makeHDRI(ANARIDevice device, HdriMap map = {})
{
  const uint32_t W = map.size[0], H = map.size[1];
  std::vector<vec3> texels(W * H);
  for (uint32_t y = 0; y < H; ++y) {
    const float value = y < H / 2 ? map.upperRadiance : map.lowerRadiance;
    for (uint32_t x = 0; x < W; ++x)
      texels[y * W + x] = vec3{value, value, value};
  }
  auto radiance = anari::newArray2D(device, ANARI_FLOAT32_VEC3, W, H);
  std::memcpy(anari::map<vec3>(device, radiance),
      texels.data(),
      texels.size() * sizeof(vec3));
  anari::unmap(device, radiance);

  auto light = anari::newObject<anari::Light>(device, "hdri");
  anari::setParameter(device, light, "direction", vec3{0.f, 0.f, 1.f});
  // HDRI row zero faces -up, so the first half illuminates the +Y plane.
  anari::setParameter(device, light, "up", vec3{0.f, -1.f, 0.f});
  anari::setParameter(device, light, "scale", 1.f);
  anari::setParameter(device, light, "visible", map.visible);
  anari::setAndReleaseParameter(device, light, "radiance", radiance);
  anari::commitParameters(device, light);
  return light;
}

static anari::Light makeDirectional(ANARIDevice device)
{
  auto light = anari::newObject<anari::Light>(device, "directional");
  anari::setParameter(device, light, "direction", vec3{0.3f, -1.f, 0.2f});
  anari::setParameter(device, light, "irradiance", 2.f);
  anari::commitParameters(device, light);
  return light;
}

static double render(ANARIDevice device, bool hdri, bool directional)
{
  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  const vec3 center = {0.f, -1000.f, 0.f};
  anari::setParameterArray1D(device, geometry, "vertex.position", &center, 1);
  anari::setParameter(device, geometry, "radius", 1000.f);
  anari::commitParameters(device, geometry);

  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", vec3{0.8f, 0.8f, 0.8f});
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  std::vector<anari::Light> lights;
  if (hdri)
    lights.push_back(makeHDRI(device));
  if (directional)
    lights.push_back(makeDirectional(device));

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  if (!lights.empty())
    anari::setParameterArray1D(
        device, world, "light", lights.data(), lights.size());
  anari::release(device, surface);
  for (auto l : lights)
    anari::release(device, l);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 2.f, 0.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, -0.25f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  // Non-visible background so the measured region is only the lit ground, not
  // the directly-seen environment (which is not part of the linearity check).
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", PIXEL_SAMPLES);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", IMAGE_SIZE);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);
  auto fb = anari::map<vec4>(device, frame, "channel.color");

  double sum = 0.0;
  uint64_t n = 0;
  for (uint32_t y = IMAGE_SIZE[1] / 8; y < IMAGE_SIZE[1] / 2; ++y) {
    for (uint32_t x = IMAGE_SIZE[0] / 8; x < 7 * IMAGE_SIZE[0] / 8; ++x) {
      const vec4 &p = fb.data[y * IMAGE_SIZE[0] + x];
      sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
      ++n;
    }
  }
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return n ? sum / double(n) : 0.0;
}

// A planar Lambertian under a uniform environment of radiance L reflects ρL
// exactly (view-independent). Used to catch two-strategy MIS energy error
// (double-count, or cosine NEE omitted from the env-CDF weight). `pbr` uses
// physicallyBased with specular=0 so the continuation lobe has a finite pdf
// — the miss-side three-way weight is invisible to matte (pdf=0).
static double renderDiffusePlane(ANARIDevice device,
    bool pbr,
    const std::vector<HdriMap> &maps = {HdriMap{}},
    bool backdrop = false)
{
  const std::array<vec3, 4> pos = {vec3{-20.f, 0.f, -20.f},
      vec3{20.f, 0.f, -20.f},
      vec3{20.f, 0.f, 20.f},
      vec3{-20.f, 0.f, 20.f}};
  // Winding produces +Y geometric normals so the camera above the plane sees
  // front faces (v0-v2-v1: e1×e2 = +Y).
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 2, 1}, std::array<unsigned, 3>{0, 3, 2}};

  auto geometry = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(
      device, geometry, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(
      device, geometry, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geometry);

  anari::Material material;
  if (pbr) {
    material = anari::newObject<anari::Material>(device, "physicallyBased");
    anari::setParameter(device, material, "baseColor", vec3{0.8f, 0.8f, 0.8f});
    anari::setParameter(device, material, "metallic", 0.f);
    anari::setParameter(device, material, "roughness", 1.f);
    anari::setParameter(device, material, "specular", 0.f);
  } else {
    material = anari::newObject<anari::Material>(device, "matte");
    anari::setParameter(device, material, "color", vec3{0.8f, 0.8f, 0.8f});
  }
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  std::vector<anari::Light> lights;
  for (const auto &map : maps)
    lights.push_back(makeHDRI(device, map));

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::setParameterArray1D(
      device, world, "light", lights.data(), lights.size());
  anari::release(device, surface);
  for (auto light : lights)
    anari::release(device, light);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 4.f, 0.f});
  anari::setParameter(
      device, camera, "direction", vec3{0.f, backdrop ? 1.f : -1.f, 0.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 0.f, 1.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", PIXEL_SAMPLES);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", IMAGE_SIZE);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);
  auto fb = anari::map<vec4>(device, frame, "channel.color");

  double sum = 0.0;
  uint64_t n = 0;
  for (uint32_t y = IMAGE_SIZE[1] / 4; y < 3 * IMAGE_SIZE[1] / 4; ++y) {
    for (uint32_t x = IMAGE_SIZE[0] / 4; x < 3 * IMAGE_SIZE[0] / 4; ++x) {
      const vec4 &p = fb.data[y * IMAGE_SIZE[0] + x];
      sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
      ++n;
    }
  }
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return n ? sum / double(n) : 0.0;
}

static bool checkEnergy(const char *name, double actual, double expected)
{
  printf("%s: actual=%f expected=%f\n", name, actual, expected);
  const double tolerance = expected > 0.0 ? 0.03 * expected : 1e-4;
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    fprintf(stderr, "FAIL: %s (tolerance=%f)\n", name, tolerance);
    return false;
  }
  return true;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  const double both = render(device, true, true);
  const double env = render(device, true, false);
  const double sun = render(device, false, true);
  const double plane = renderDiffusePlane(device, false);
  const double planePbr = renderDiffusePlane(device, true);
  // The sum of constant environments is constant: rho * (0.4 + 0.2) = 0.48.
  // Unequal powers must form a mixture, not a sum of normalized densities.
  bool passed = true;
  for (bool pbr : {false, true}) {
    passed &= checkEnergy(pbr ? "multiple HDRIs PBR" : "multiple HDRIs matte",
        renderDiffusePlane(device, pbr, {{0.4f, 0.4f}, {0.2f, 0.2f}}),
        0.48);
    // Even a one-texel map is a constant environment. Its CDF jitters theta
    // uniformly, so its solid-angle density is not uniform on the sphere.
    passed &= checkEnergy(pbr ? "one-texel HDRI PBR" : "one-texel HDRI matte",
        renderDiffusePlane(device, pbr, {{0.6f, 0.6f, true, {1, 1}}}),
        0.48);
    // Only the upper hemisphere illuminates the plane, regardless of how
    // bright the folded-away hemisphere is. Filtering is confined to a narrow
    // band at the horizon where the cosine factor vanishes.
    passed &= checkEnergy(pbr ? "nonuniform HDRI PBR" : "nonuniform HDRI matte",
        renderDiffusePlane(device, pbr, {{0.6f, 6.f}}),
        0.48);
    passed &= checkEnergy(
        pbr ? "unequal HDRI mixture PBR" : "unequal HDRI mixture matte",
        renderDiffusePlane(device, pbr, {{0.4f, 0.04f}, {0.2f, 2.f}}),
        0.48);
    // Hiding a light's background does not remove its illumination.
    passed &= checkEnergy(pbr ? "hidden HDRI PBR" : "hidden HDRI matte",
        renderDiffusePlane(device, pbr, {{0.6f, 0.6f, false}}),
        0.48);
    passed &= checkEnergy(
        pbr ? "visible and hidden HDRIs PBR" : "visible and hidden HDRIs matte",
        renderDiffusePlane(device, pbr, {{0.4f, 0.4f}, {0.2f, 0.2f, false}}),
        0.48);
    passed &= checkEnergy(pbr ? "black HDRI PBR" : "black HDRI matte",
        renderDiffusePlane(device, pbr, {{0.f, 0.f}}),
        0.0);
  }
  passed &= checkEnergy("visible backdrop",
      renderDiffusePlane(device, false, {{0.6f, 0.6f}}, true),
      0.6);
  passed &= checkEnergy("hidden backdrop",
      renderDiffusePlane(device, false, {{0.6f, 0.6f, false}}, true),
      0.0);
  anari::release(device, device);
  if (!passed)
    return 1;

  const double sum = env + sun;
  const double relErr = sum > 0.0 ? std::abs(both - sum) / sum : 1.0;
  printf("both=%f  env=%f  sun=%f  sum=%f  relErr=%f\n",
      both,
      env,
      sun,
      sum,
      relErr);

  if (env <= 0.0) {
    fprintf(stderr, "FAIL: HDRI environment did not light the ground\n");
    return 1;
  }
  // Lambertian under a uniform environment of radiance L reflects ρL. The HDRI
  // texels are 0.6 and the matte albedo is 0.8, so the ground mean must match
  // 0.48 — a broken two-strategy MIS (double-count, or cosine NEE omitted from
  // the env-CDF weight) shows up as a mean energy error, not just extra noise.
  constexpr double albedo = 0.8;
  constexpr double envRadiance = 0.6;
  const double expected = albedo * envRadiance;
  const double relErrEnv =
      expected > 0.0 ? std::abs(plane - expected) / expected : 1.0;
  printf("plane=%f  planePbr=%f  envExpected=%f  relErrEnv=%f\n",
      plane,
      planePbr,
      expected,
      relErrEnv);
  constexpr double ENV_ENERGY_TOLERANCE = 0.05;
  if (!(relErrEnv <= ENV_ENERGY_TOLERANCE)) {
    fprintf(stderr,
        "FAIL: matte plane under uniform HDRI not ρL (plane=%f expected=%f "
        "relErr=%f, tol %f)\n",
        plane,
        expected,
        relErrEnv,
        ENV_ENERGY_TOLERANCE);
    return 1;
  }
  const double relErrPbr =
      expected > 0.0 ? std::abs(planePbr - expected) / expected : 1.0;
  printf("relErrPbr=%f\n", relErrPbr);
  if (!(relErrPbr <= ENV_ENERGY_TOLERANCE)) {
    fprintf(stderr,
        "FAIL: PBR plane under uniform HDRI not ρL (planePbr=%f expected=%f "
        "relErr=%f, tol %f) — miss-side env MIS likely omitted p_C\n",
        planePbr,
        expected,
        relErrPbr,
        ENV_ENERGY_TOLERANCE);
    return 1;
  }
  constexpr double TOLERANCE = 0.03;
  if (!(relErr <= TOLERANCE)) {
    fprintf(stderr,
        "FAIL: HDRI + directional not additive (relErr=%f, tol %f): env-MIS "
        "pick-probability fold is biased\n",
        relErr,
        TOLERANCE);
    return 1;
  }
  printf("power light pick env-MIS additivity passed\n");
  return 0;
}
