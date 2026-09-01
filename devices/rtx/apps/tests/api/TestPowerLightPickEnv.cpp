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

// Power-proportional Light Pick must keep the environment (HDRI) MIS fold
// unbiased. The HDRI is reached by env-CDF NEE, cosine-hemisphere NEE, and the
// BSDF escape; power picking multiplies its pick probability into the env light
// density on every NEE-side weight. Light transport is linear, so a scene lit
// by an HDRI plus a directional light must equal the sum of the two
// single-light renders. A second check (matte plane under a uniform HDRI) must
// match ρL — a broken two-strategy partition (double-count, or cosine omitted
// from the env-CDF weight) shows up as a mean energy error. Rendered with
// 'quality' into a linear float buffer, firefly off.

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR)
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
}

static constexpr uvec2 IMAGE_SIZE = {256, 256};
static constexpr int PIXEL_SAMPLES = 256;

static anari::Light makeHDRI(ANARIDevice device)
{
  // A uniform (constant-radiance) environment.
  constexpr uint32_t W = 8, H = 4;
  std::vector<vec3> texels(W * H, vec3{0.6f, 0.6f, 0.6f});
  auto radiance = anari::newArray2D(device, ANARI_FLOAT32_VEC3, W, H);
  std::memcpy(anari::map<vec3>(device, radiance),
      texels.data(),
      texels.size() * sizeof(vec3));
  anari::unmap(device, radiance);

  auto light = anari::newObject<anari::Light>(device, "hdri");
  anari::setParameter(device, light, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, light, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(device, light, "scale", 1.f);
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
static double renderDiffusePlane(ANARIDevice device, bool pbr)
{
  const std::array<vec3, 4> pos = {vec3{-20.f, 0.f, -20.f},
      vec3{20.f, 0.f, -20.f},
      vec3{20.f, 0.f, 20.f},
      vec3{-20.f, 0.f, 20.f}};
  // Winding produces +Y geometric normals so the camera above the plane sees
  // front faces (v0-v1-v2: e1×e2 = +Y).
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

  auto light = makeHDRI(device);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::setParameterArray1D(device, world, "light", &light, 1);
  anari::release(device, surface);
  anari::release(device, light);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 4.f, 0.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, -1.f, 0.f});
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

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  const double both = render(device, true, true);
  const double env = render(device, true, false);
  const double sun = render(device, false, true);
  const double plane = renderDiffusePlane(device, false);
  const double planePbr = renderDiffusePlane(device, true);
  anari::release(device, device);

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
