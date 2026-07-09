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

// Geometry Light sampling must be exact under a non-uniform instance scale. An
// emissive quad baked at a 2x1 world footprint (identity instance) must light a
// floor identically to a unit 1x1 emissive quad stretched to the same footprint
// by a scale(2,1,1) instance transform. The two describe the same world emitter,
// so any pixel difference means the per-primitive area Jacobian in the sampler /
// hit-side pdf is wrong — the silent bias non-uniform scale would otherwise
// introduce. Rendered with 'quality', linear float buffer, firefly filter off.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
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
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using mat4 = std::array<float, 16>;

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
static constexpr float EMISSIVE_RADIANCE = 8.f;
static constexpr float QUAD_Y = 1.5f;

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
  anari::setParameter(device, mat, "roughness", 1.f);
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// A down-facing emissive quad with the given half-extents in X and Z.
static anari::Surface makeEmissiveQuad(ANARIDevice device, float hx, float hz)
{
  const std::array<vec3, 4> pos = {vec3{-hx, QUAD_Y, -hz},
      vec3{hx, QUAD_Y, -hz},
      vec3{hx, QUAD_Y, hz},
      vec3{-hx, QUAD_Y, hz}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(device,
      mat,
      "emissive",
      vec3{EMISSIVE_RADIANCE, EMISSIVE_RADIANCE, EMISSIVE_RADIANCE});
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// Render the floor plus an emitter. `scaled` selects a unit quad stretched to a
// 2x1 footprint by a scale(2,1,1) instance; otherwise a baked 2x1 quad with an
// identity instance. Both yield the same world-space emitter.
static std::vector<vec4> render(ANARIDevice device, bool scaled)
{
  auto floor = makeFloor(device);
  auto emitter =
      scaled ? makeEmissiveQuad(device, 1.f, 1.f) : makeEmissiveQuad(device, 2.f, 1.f);

  auto floorGroup = anari::newObject<anari::Group>(device);
  anari::setParameterArray1D(device, floorGroup, "surface", &floor, 1);
  anari::release(device, floor);
  anari::commitParameters(device, floorGroup);

  auto emitterGroup = anari::newObject<anari::Group>(device);
  anari::setParameterArray1D(device, emitterGroup, "surface", &emitter, 1);
  anari::release(device, emitter);
  anari::commitParameters(device, emitterGroup);

  auto floorInstance = anari::newObject<anari::Instance>(device, "transform");
  anari::setAndReleaseParameter(device, floorInstance, "group", floorGroup);
  anari::commitParameters(device, floorInstance);

  auto emitterInstance = anari::newObject<anari::Instance>(device, "transform");
  anari::setAndReleaseParameter(device, emitterInstance, "group", emitterGroup);
  if (scaled) {
    // Column-major scale(2,1,1).
    const mat4 xfm = {2.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f};
    anari::setParameter(
        device, emitterInstance, "transform", ANARI_FLOAT32_MAT4, xfm.data());
  }
  anari::commitParameters(device, emitterInstance);

  const std::array<anari::Instance, 2> instances = {floorInstance, emitterInstance};

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "instance", instances.data(), instances.size());
  anari::release(device, floorInstance);
  anari::release(device, emitterInstance);
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
  std::vector<vec4> out(fb.data, fb.data + size_t(fb.width) * fb.height);
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return out;
}

static double poolMeanLuminance(const std::vector<vec4> &fb)
{
  double sum = 0.0;
  uint64_t n = 0;
  for (uint32_t y = IMAGE_SIZE[1] / 8; y < IMAGE_SIZE[1] / 2; ++y) {
    for (uint32_t x = 3 * IMAGE_SIZE[0] / 8; x < 5 * IMAGE_SIZE[0] / 8; ++x) {
      const vec4 &p = fb[y * IMAGE_SIZE[0] + x];
      sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
      ++n;
    }
  }
  return n ? sum / double(n) : 0.0;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const double baked = poolMeanLuminance(render(device, false));
  const double scaled = poolMeanLuminance(render(device, true));

  anari::release(device, device);

  const double relErr =
      baked > 0.0 ? std::abs(scaled - baked) / baked : (scaled > 0 ? 1.0 : 0.0);
  printf("baked=%f  scaledInstance=%f  relErr=%f\n", baked, scaled, relErr);

  if (baked <= 0.0 || scaled <= 0.0) {
    fprintf(stderr, "FAIL: floor pool unlit (baked=%f scaled=%f)\n", baked, scaled);
    return 1;
  }
  constexpr double TOLERANCE = 0.05;
  if (relErr > TOLERANCE) {
    fprintf(stderr,
        "FAIL: non-uniform instance scale biased the emitter (relErr=%f > %f)\n",
        relErr,
        TOLERANCE);
    return 1;
  }
  printf("emissive instance-scale Jacobian passed\n");
  return 0;
}
