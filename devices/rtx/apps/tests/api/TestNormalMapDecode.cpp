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

// physicallyBased `normal` sampler semantics, per the ANARI spec: the sampler
// returns the tangent-space normal itself; decoding texels (2 * texel - 1) is
// the app's job, via the sampler's outTransform/outOffset. Pins: a flat normal
// map (every texel (0.5, 0.5, 1)) with the spec's decode lights a floor exactly
// like no normal map, and the same map WITHOUT the decode does not (the device
// must not decode again). Runs against `physicallyBased`, and additionally
// against the always-MDL `physicallyBasedMDL` subtype when MDL support is
// compiled in. Linear float buffer, firefly off.

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
using vec2 = std::array<float, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using mat4 = std::array<float, 16>; // column-major

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
static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;

enum class NormalMap
{
  NONE,
  FLAT_SPEC_DECODE, // (0.5, 0.5, 1) texels, 2 * texel - 1 in outTransform
  FLAT_RAW, // (0.5, 0.5, 1) texels, no decode: a tilted normal per the spec
};

// The receiver under test: a diffuse floor with texture coordinates (so a
// tangent frame is generated) whose `normal` input is the experiment variable.
static anari::Surface makeFloor(
    ANARIDevice device, const char *subtype, NormalMap normalMap)
{
  const std::array<vec3, 4> pos = {vec3{-6.f, 0.f, -6.f},
      vec3{6.f, 0.f, -6.f},
      vec3{6.f, 0.f, 6.f},
      vec3{-6.f, 0.f, 6.f}};
  const std::array<vec2, 4> uv = {
      vec2{0.f, 0.f}, vec2{1.f, 0.f}, vec2{1.f, 1.f}, vec2{0.f, 1.f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "vertex.attribute0", uv.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, subtype);
  anari::setParameter(device, mat, "baseColor", vec3{0.6f, 0.6f, 0.6f});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  if (normalMap != NormalMap::NONE) {
    auto sampler = anari::newObject<anari::Sampler>(device, "image2D");
    anari::setParameter(device, sampler, "inAttribute", "attribute0");
    anari::setParameter(device, sampler, "filter", "nearest");
    const std::array<vec3, 4> img = {vec3{0.5f, 0.5f, 1.f},
        vec3{0.5f, 0.5f, 1.f},
        vec3{0.5f, 0.5f, 1.f},
        vec3{0.5f, 0.5f, 1.f}};
    anari::setParameterArray2D(device, sampler, "image", img.data(), 2, 2);
    if (normalMap == NormalMap::FLAT_SPEC_DECODE) {
      const mat4 outTransform = {2.f,
          0.f,
          0.f,
          0.f,
          0.f,
          2.f,
          0.f,
          0.f,
          0.f,
          0.f,
          2.f,
          0.f,
          0.f,
          0.f,
          0.f,
          1.f};
      anari::setParameter(device,
          sampler,
          "outTransform",
          ANARI_FLOAT32_MAT4,
          outTransform.data());
      anari::setParameter(
          device, sampler, "outOffset", vec4{-1.f, -1.f, -1.f, 0.f});
    }
    anari::commitParameters(device, sampler);
    anari::setAndReleaseParameter(device, mat, "normal", sampler);
  }
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// Constant emissive quad straight above the floor, so a tilted shading normal
// darkens the pool beneath it.
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

static double poolMean(
    ANARIDevice device, const char *subtype, NormalMap normalMap)
{
  const std::array<anari::Surface, 2> surfaces = {
      makeFloor(device, subtype, normalMap), makeEmitter(device)};
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

static bool checkPool(const char *label, double value)
{
  if (!std::isfinite(value) || value <= 0.0) {
    fprintf(stderr, "FAIL: %s produced a dark or non-finite pool\n", label);
    return false;
  }
  return true;
}

// A flat map with the spec's decode is the unperturbed normal.
static bool checkEqual(const char *label, double mapped, double none)
{
  const double relErr = std::abs(mapped - none) / none;
  printf("%s: %f vs %f relErr=%f\n", label, mapped, none, relErr);
  if (relErr > 0.02) {
    fprintf(stderr,
        "FAIL: %s differs from no normal map (relErr=%f > 0.02) — the "
        "sampler's value is being decoded again\n",
        label,
        relErr);
    return false;
  }
  return true;
}

// Raw (0.5, 0.5, 1) is a normal tilted ~35° off the surface normal, which
// must darken the pool. Matching no map means the device decoded it itself.
static bool checkDiffers(const char *label, double raw, double none)
{
  const double relErr = std::abs(raw - none) / none;
  printf("%s (must differ): %f vs %f relErr=%f\n", label, raw, none, relErr);
  if (relErr < 0.05) {
    fprintf(stderr,
        "FAIL: %s raw texels shade like no normal map (relErr=%f < 0.05) — "
        "the device decodes the sampler's value\n",
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
    const double none = poolMean(device, subtype, NormalMap::NONE);
    const double flat = poolMean(device, subtype, NormalMap::FLAT_SPEC_DECODE);
    const double raw = poolMean(device, subtype, NormalMap::FLAT_RAW);
    char label[96];
    snprintf(label, sizeof(label), "%s/flat-spec-decode", subtype);
    ok = checkPool(label, none) && checkPool(label, flat)
        && checkEqual(label, flat, none) && ok;
    snprintf(label, sizeof(label), "%s/flat-raw", subtype);
    ok = checkPool(label, raw) && checkDiffers(label, raw, none) && ok;
  }

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("normal map decode semantics passed\n");
  return 0;
}
