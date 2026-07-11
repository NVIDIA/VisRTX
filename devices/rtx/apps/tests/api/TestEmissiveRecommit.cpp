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

// A constant-emissive surface must STAY a Geometry Light across re-commits that
// don't touch `emissive`. The MDL-backed PhysicallyBased material captured the
// pre-translate `emissive` key, which the ANARI->MDL translation consumes at
// first commit — so an unrelated edit (here: roughness) + re-commit zeroed the
// capture and silently de-registered the light while the surface kept glowing
// on path hits. The interactive renderer is NEE-only, so the floor going dark
// after the re-commit is the regression signal. Runs ungated: native PBR has no
// translation and must trivially pass. Linear float, firefly off.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <cmath>
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
static constexpr int PIXEL_SAMPLES = 256;
static constexpr float RADIANCE = 8.f;
// Same frame re-rendered twice is deterministic; any drift means the light set
// changed under us.
static constexpr double REL_TOL = 0.02;
static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;

static anari::Surface makeFloor(ANARIDevice device)
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

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.6f, 0.6f, 0.6f});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// Down-facing constant Emissive Surface of the given material subtype; the
// material handle is returned (not released) so the test can edit and
// re-commit it.
static anari::Surface makeEmissiveQuad(
    ANARIDevice device, const char *subtype, anari::Material *outMat)
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

  auto mat = anari::newObject<anari::Material>(device, subtype);
  anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  anari::setParameter(
      device, mat, "emissive", vec3{RADIANCE, RADIANCE, RADIANCE});
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  *outMat = mat;
  return surface;
}

static double renderPoolMean(ANARIDevice device, anari::Frame frame)
{
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
  return n ? sum / double(n) : 0.0;
}

static bool checkSubtype(ANARIDevice device, const char *subtype)
{
  anari::Material emissiveMat{};
  const std::array<anari::Surface, 2> surfaces = {
      makeFloor(device), makeEmissiveQuad(device, subtype, &emissiveMat)};

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

  // Interactive is NEE-only: if the Geometry Light drops, the floor goes dark.
  auto renderer = anari::newObject<anari::Renderer>(device, "interactive");
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

  const double before = renderPoolMean(device, frame);

  // Unrelated edit: `emissive` is untouched, so the light must survive.
  anari::setParameter(device, emissiveMat, "roughness", 0.5f);
  anari::commitParameters(device, emissiveMat);

  const double after = renderPoolMean(device, frame);
  anari::release(device, emissiveMat);
  anari::release(device, frame);

  const double relErr = before > 0.0 ? std::abs(after - before) / before
                                     : (after > 0 ? 1.0 : 0.0);
  printf(
      "%s: before=%f  after=%f  relErr=%f\n", subtype, before, after, relErr);

  // A NaN pool would sail through the sign checks below.
  if (!std::isfinite(before) || !std::isfinite(after)) {
    fprintf(stderr, "FAIL: %s produced a non-finite pool mean\n", subtype);
    return false;
  }
  if (before <= 0.0) {
    fprintf(stderr, "FAIL: %s floor unlit before the re-commit\n", subtype);
    return false;
  }
  if (after <= 0.0) {
    fprintf(stderr,
        "FAIL: %s: re-committing an unrelated parameter dropped the Geometry "
        "Light (floor went dark)\n",
        subtype);
    return false;
  }
  if (relErr > REL_TOL) {
    fprintf(stderr,
        "FAIL: %s: emission changed across an unrelated re-commit "
        "(relErr=%f)\n",
        subtype,
        relErr);
    return false;
  }
  return true;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  bool ok = checkSubtype(device, "physicallyBased");
#ifdef VISRTX_TEST_MDL_WRAPPER
  // `physicallyBased` maps to the MDL-backed wrapper only when the device is
  // built with USE_MDL_FOR_PHYSICALLY_BASED; the always-exposed
  // `physicallyBasedMDL` subtype pins the wrapper's capture regression in
  // every MDL-enabled build.
  ok = checkSubtype(device, "physicallyBasedMDL") && ok;
#endif

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("emissive light survives unrelated re-commit\n");
  return 0;
}
