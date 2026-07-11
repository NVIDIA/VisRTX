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

// physicallyBased `specular` default semantics, per the ANARI spec: `specular`
// defaults to 0.0 (KHR_materials_specular scale), so a default dielectric floor
// is pure diffuse — no Fresnel. Pins: UNSET shades identically to explicit
// `specular = 0`, and NOT like `specular = 1` (which adds the dielectric
// Fresnel and darkens the pool). `specularColor` is inert while specular is 0,
// so UNSET also equals an explicit `specularColor = (1,1,1)`. Runs against
// `physicallyBased` on BOTH PBR backend builds, and additionally against the
// always-MDL `physicallyBasedMDL` subtype when MDL support is compiled in —
// so the module default matches the native backend even when physicallyBased
// is native. Linear float buffer, firefly off.

// anari_cpp
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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
static constexpr int PIXEL_SAMPLES = 512;
static constexpr float RADIANCE = 8.f;
static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;

enum class SpecularMode
{
  UNSET,
  EXPLICIT_ZERO, // specular = 0.f (the spec default, made explicit)
  EXPLICIT_ONE, // specular = 1.f (full dielectric Fresnel)
  EXPLICIT_COLOR, // specularColor = (1,1,1), specular untouched
};

// The receiver under test: a default-ish diffuse floor whose specular input
// is the experiment variable. `subtype` selects the material implementation
// (physicallyBased, or the always-MDL physicallyBasedMDL wrapper).
static anari::Surface makeFloor(
    ANARIDevice device, const char *subtype, SpecularMode mode)
{
  const std::array<vec3, 4> pos = {vec3{-6.f, 0.f, -6.f},
      vec3{6.f, 0.f, -6.f},
      vec3{6.f, 0.f, 6.f},
      vec3{-6.f, 0.f, 6.f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, subtype);
  anari::setParameter(device, mat, "baseColor", vec3{0.6f, 0.6f, 0.6f});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  if (mode == SpecularMode::EXPLICIT_ZERO)
    anari::setParameter(device, mat, "specular", 0.f);
  else if (mode == SpecularMode::EXPLICIT_ONE)
    anari::setParameter(device, mat, "specular", 1.f);
  else if (mode == SpecularMode::EXPLICIT_COLOR)
    anari::setParameter(device, mat, "specularColor", vec3{1.f, 1.f, 1.f});
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// Constant emissive quad lighting the floor (the parity-test geometry).
static anari::Surface makeEmitter(ANARIDevice device)
{
  const std::array<vec3, 4> pos = {vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, QUAD_HALF},
      vec3{-QUAD_HALF, QUAD_Y, QUAD_HALF}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  anari::setParameter(
      device, mat, "emissive", vec3{RADIANCE, RADIANCE, RADIANCE});
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

static double poolMean(ANARIDevice device,
    const char *subtype,
    SpecularMode mode,
    const char *rendererType)
{
  const std::array<anari::Surface, 2> surfaces = {
      makeFloor(device, subtype, mode), makeEmitter(device)};
  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(device, s);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.5f, -3.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, -0.15f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, rendererType);
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
    for (uint32_t x = 3 * IMAGE_SIZE[0] / 8; x < 5 * IMAGE_SIZE[0] / 8; ++x) {
      const vec4 &p = fb.data[y * IMAGE_SIZE[0] + x];
      sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
      ++n;
    }
  }
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return n ? sum / double(n) : 0.0;
}

static bool checkEqual(const char *label, double unset, double explicitOne)
{
  const double relErr = explicitOne > 0.0
      ? std::abs(unset - explicitOne) / explicitOne
      : (unset > 0.0 ? 1.0 : 0.0);
  printf("%s: %f vs %f relErr=%f\n", label, unset, explicitOne, relErr);
  if (!std::isfinite(unset) || !std::isfinite(explicitOne) || unset <= 0.0
      || explicitOne <= 0.0) {
    fprintf(stderr, "FAIL: %s produced a dark or non-finite pool\n", label);
    return false;
  }
  // Identical semantics: only sampling noise (and, cross-backend, residual
  // model differences well under a percent) apart.
  if (relErr > 0.02) {
    fprintf(stderr,
        "FAIL: %s not equivalent (relErr=%f > 0.02)\n",
        label,
        relErr);
    return false;
  }
  return true;
}

// Guards that the default is NOT the old full-Fresnel behavior: adding the
// dielectric Fresnel (specular = 1) must measurably darken the diffuse pool.
static bool checkDiffers(const char *label, double unset, double explicitOne)
{
  const double relErr = unset > 0.0 ? std::abs(unset - explicitOne) / unset : 0.0;
  printf("%s (must differ): %f vs %f relErr=%f\n", label, unset, explicitOne, relErr);
  if (!std::isfinite(unset) || !std::isfinite(explicitOne) || unset <= 0.0
      || explicitOne <= 0.0) {
    fprintf(stderr, "FAIL: %s produced a dark or non-finite pool\n", label);
    return false;
  }
  if (relErr < 0.05) {
    fprintf(stderr,
        "FAIL: %s default matches specular=1 (relErr=%f < 0.05) — dielectric "
        "Fresnel is on by default\n",
        label,
        relErr);
    return false;
  }
  return true;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  bool ok = true;
  const std::array<const char *, 2> subtypes = {"physicallyBased",
#ifdef VISRTX_TEST_MDL_WRAPPER
      "physicallyBasedMDL"
#else
      nullptr
#endif
  };
  for (const char *subtype : subtypes) {
    if (!subtype)
      continue;
    for (const char *renderer : {"quality", "interactive"}) {
      const double base =
          poolMean(device, subtype, SpecularMode::UNSET, renderer);
      const double zero =
          poolMean(device, subtype, SpecularMode::EXPLICIT_ZERO, renderer);
      const double one =
          poolMean(device, subtype, SpecularMode::EXPLICIT_ONE, renderer);
      const double color =
          poolMean(device, subtype, SpecularMode::EXPLICIT_COLOR, renderer);
      char label[96];
      snprintf(label, sizeof(label), "%s/%s/specular-default", subtype, renderer);
      ok = checkEqual(label, base, zero) && ok;
      snprintf(label, sizeof(label), "%s/%s/specularColor-inert", subtype, renderer);
      ok = checkEqual(label, base, color) && ok;
      snprintf(label, sizeof(label), "%s/%s/not-full-fresnel", subtype, renderer);
      ok = checkDiffers(label, base, one) && ok;
    }
  }

#ifdef VISRTX_TEST_MDL_WRAPPER
  // Cross-backend parity on ONE build for the spec default: an UNSET-specular
  // floor is pure diffuse, so the MDL wrapper and the native reference must
  // agree (the dielectric layer is off, sidestepping the residual specular>0
  // BRDF gap tracked separately). Trivially equal when physicallyBased IS the
  // wrapper.
  for (const char *renderer : {"quality", "interactive"}) {
    const double nat =
        poolMean(device, "physicallyBased", SpecularMode::UNSET, renderer);
    const double wrap =
        poolMean(device, "physicallyBasedMDL", SpecularMode::UNSET, renderer);
    char label[96];
    snprintf(label, sizeof(label), "%s/wrapper-vs-native", renderer);
    ok = checkEqual(label, wrap, nat) && ok;
  }
#endif

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("pbr specular default semantics passed\n");
  return 0;
}
