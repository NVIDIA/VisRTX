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

// One renderer, several MDL material-library updates: swapping the world to
// one whose `mdl` material compiles a DIFFERENT source forces the renderer to
// rebuild its OptiX pipeline mid-life (the MDL registry timestamp advances).
// The rebuild must release the previous pipeline, program groups and MDL
// modules and produce a working replacement — this pins the rebuild/teardown
// path no other test exercises (their renderers build exactly once). Each
// swap must still light the floor; the process must exit cleanly.

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
#include <string>

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

static constexpr uvec2 IMAGE_SIZE = {128, 128};
static constexpr int PIXEL_SAMPLES = 64;
static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;
static constexpr int REBUILD_CYCLES = 4;

// Distinct intensity per cycle => distinct source hash => the old material's
// registry slot is released and a new one compiled: a library update the
// renderer must absorb by rebuilding its pipeline.
static std::string mdlSource(int cycle)
{
  return std::string(R"mdl(mdl 1.6;
import ::df::*;
export material emissive() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color()mdl")
      + std::to_string(8 + cycle) + R"mdl(.0))));
)mdl";
}

static anari::World makeWorld(ANARIDevice device, int cycle)
{
  const std::array<vec3, 4> floorPos = {vec3{-6.f, 0.f, -6.f},
      vec3{6.f, 0.f, -6.f},
      vec3{6.f, 0.f, 6.f},
      vec3{-6.f, 0.f, 6.f}};
  const std::array<vec3, 4> quadPos = {vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, QUAD_HALF},
      vec3{-QUAD_HALF, QUAD_Y, QUAD_HALF}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto floorGeom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(
      device, floorGeom, "vertex.position", floorPos.data(), 4);
  anari::setParameterArray1D(
      device, floorGeom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, floorGeom);

  auto floorMat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, floorMat, "baseColor", vec3{0.6f, 0.6f, 0.6f});
  anari::setParameter(device, floorMat, "roughness", 1.f);
  anari::commitParameters(device, floorMat);

  auto floor = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, floor, "geometry", floorGeom);
  anari::setAndReleaseParameter(device, floor, "material", floorMat);
  anari::commitParameters(device, floor);

  auto quadGeom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(
      device, quadGeom, "vertex.position", quadPos.data(), 4);
  anari::setParameterArray1D(device, quadGeom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, quadGeom);

  auto mat = anari::newObject<anari::Material>(device, "mdl");
  anari::setParameter(device, mat, "sourceType", "code");
  anari::setParameter(device, mat, "source", mdlSource(cycle));
  anari::setParameter(device, mat, "materialName", "emissive");
  anari::commitParameters(device, mat);

  auto quad = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, quad, "geometry", quadGeom);
  anari::setAndReleaseParameter(device, quad, "material", mat);
  anari::commitParameters(device, quad);

  const std::array<anari::Surface, 2> surfaces = {floor, quad};
  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(device, s);
  anari::commitParameters(device, world);
  return world;
}

static double poolMean(ANARIDevice device, anari::Frame frame)
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

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

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
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);

  bool ok = true;
  double previous = -1.0;
  for (int cycle = 0; cycle < REBUILD_CYCLES; ++cycle) {
    auto world = makeWorld(device, cycle);
    anari::setAndReleaseParameter(device, frame, "world", world);
    anari::commitParameters(device, frame);

    const double pool = poolMean(device, frame);
    printf("cycle %d: pool=%f\n", cycle, pool);
    if (!std::isfinite(pool) || pool <= 0.0) {
      fprintf(stderr, "FAIL: cycle %d rendered dark after rebuild\n", cycle);
      ok = false;
      break;
    }
    // Each cycle's intensity is strictly larger: the pool must grow, proving
    // the rebuilt pipeline actually picked up the NEW material.
    if (pool <= previous) {
      fprintf(stderr,
          "FAIL: cycle %d pool %f did not increase over %f — stale pipeline?\n",
          cycle,
          pool,
          previous);
      ok = false;
      break;
    }
    previous = pool;
  }

  anari::release(device, frame);
  anari::release(device, device);

  if (!ok)
    return 1;
  printf("mdl pipeline rebuild passed\n");
  return 0;
}
