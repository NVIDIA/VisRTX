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

// Many DISTINCT `mdl` materials committed together and compiled in ONE flush:
// this is the batch the coordinator+pool loading (ADR 0009) fans out across
// worker threads. The test pins the batch semantics that parallelism must not
// break, and is meant to run at every VISRTX_MDL_COMPILE_THREADS setting:
//   1. N distinct sources compiled in one flush produce exactly N registry
//      slots and a lit render (no compile is dropped or corrupted).
//   2. N distinct + N duplicate sources in one flush still produce N slots:
//      concurrent requests for the same material dedup, never double-register.
//   3. A broken source in the batch falls back without crashing and leaves its
//      neighbours' compiles intact.
// Passing serially establishes the golden behavior; the same assertions gate
// the parallel path. Run under `VISRTX_MDL_COMPILE_THREADS=<n>` in CI to
// exercise the fan-out.

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
#include <string>
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

static constexpr uvec2 IMAGE_SIZE = {64, 64};
static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;
// Enough distinct materials that a fan-out actually overlaps compiles.
static constexpr int DISTINCT_MATERIALS = 24;

// Distinct intensity per index => distinct source hash => distinct registry
// slot / compile job.
static std::string emissiveSource(int index)
{
  return std::string(R"mdl(mdl 1.6;
import ::df::*;
export material emissive() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color()mdl")
      + std::to_string(1 + index) + R"mdl(.0))));
)mdl";
}

// Syntactically broken: compilation fails and the device falls back to its
// default material. Must not crash the batch.
static std::string brokenSource()
{
  return "mdl 1.6;\nexport material emissive() = this is not valid mdl;\n";
}

static anari::Surface makeMatteFloor(ANARIDevice device)
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

  // Matte is native in every config: it never holds a registry slot.
  auto mat = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, mat, "color", vec3{0.6f, 0.6f, 0.6f});
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

static anari::Surface makeMdlQuad(ANARIDevice device, const std::string &source)
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

  auto mat = anari::newObject<anari::Material>(device, "mdl");
  anari::setParameter(device, mat, "sourceType", "code");
  anari::setParameter(device, mat, "source", source);
  anari::setParameter(device, mat, "materialName", "emissive");
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// One mdl quad per source, all committed before the returned world commits, so
// the whole set compiles in a single flush.
static anari::World makeWorld(
    ANARIDevice device, const std::vector<std::string> &sources)
{
  std::vector<anari::Surface> surfaces;
  surfaces.push_back(makeMatteFloor(device));
  for (const auto &source : sources)
    surfaces.push_back(makeMdlQuad(device, source));

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(device, s);
  anari::commitParameters(device, world);
  return world;
}

static uint32_t queryCount(ANARIDevice device)
{
  uint32_t count = ~0u;
  if (!anari::getProperty(
          device, device, "numRegisteredMdlMaterials", count, ANARI_WAIT)) {
    fprintf(stderr, "FAIL: device has no numRegisteredMdlMaterials property\n");
    std::exit(1);
  }
  return count;
}

static double renderPoolMean(ANARIDevice device, anari::Frame frame)
{
  anari::render(device, frame);
  anari::wait(device, frame);

  auto fb = anari::map<vec4>(device, frame, "channel.color");
  double sum = 0.0;
  uint64_t n = 0;
  for (uint32_t y = 0; y < IMAGE_SIZE[1]; ++y) {
    for (uint32_t x = 0; x < IMAGE_SIZE[0]; ++x) {
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
  anari::setParameter(device, renderer, "pixelSamples", 16);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", IMAGE_SIZE);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);
  anari::release(device, camera);
  anari::release(device, renderer);

  bool ok = true;

  // Baseline AFTER a first render, so it reflects the initialized device (the
  // pre-init registry is null and reports zero).
  {
    auto world = makeWorld(device, {});
    anari::setAndReleaseParameter(device, frame, "world", world);
    anari::commitParameters(device, frame);
    renderPoolMean(device, frame);
  }
  const uint32_t baseline = queryCount(device);
  printf("baseline numRegisteredMdlMaterials = %u\n", baseline);

  // Phase 1: N distinct materials, one flush => exactly N slots + lit render.
  {
    std::vector<std::string> sources;
    for (int i = 0; i < DISTINCT_MATERIALS; ++i)
      sources.push_back(emissiveSource(i));
    auto world = makeWorld(device, sources);
    anari::setAndReleaseParameter(device, frame, "world", world);
    anari::commitParameters(device, frame);

    const double pool = renderPoolMean(device, frame);
    const uint32_t count = queryCount(device);
    printf("phase1: count=%u (expected %u), pool=%f\n",
        count,
        baseline + DISTINCT_MATERIALS,
        pool);
    if (count != baseline + DISTINCT_MATERIALS) {
      fprintf(stderr, "FAIL: phase1 expected %u distinct slots, got %u\n",
          baseline + DISTINCT_MATERIALS, count - baseline);
      ok = false;
    }
    if (!std::isfinite(pool) || pool <= 0.0) {
      fprintf(stderr, "FAIL: phase1 rendered dark\n");
      ok = false;
    }
  }

  // Phase 2: N distinct + N duplicate sources in one flush => still N slots.
  if (ok) {
    std::vector<std::string> sources;
    for (int i = 0; i < DISTINCT_MATERIALS; ++i)
      sources.push_back(emissiveSource(i));
    for (int i = 0; i < DISTINCT_MATERIALS; ++i)
      sources.push_back(emissiveSource(i)); // duplicates
    auto world = makeWorld(device, sources);
    anari::setAndReleaseParameter(device, frame, "world", world);
    anari::commitParameters(device, frame);

    const double pool = renderPoolMean(device, frame);
    const uint32_t count = queryCount(device);
    printf("phase2: count=%u (expected %u, dedup), pool=%f\n",
        count,
        baseline + DISTINCT_MATERIALS,
        pool);
    if (count != baseline + DISTINCT_MATERIALS) {
      fprintf(stderr,
          "FAIL: phase2 duplicates did not dedup: expected %u slots, got %u\n",
          baseline + DISTINCT_MATERIALS, count - baseline);
      ok = false;
    }
    if (!std::isfinite(pool) || pool <= 0.0) {
      fprintf(stderr, "FAIL: phase2 rendered dark\n");
      ok = false;
    }
  }

  // Phase 3: a broken source among valid ones falls back without crashing and
  // does not corrupt its neighbours (the render still lights).
  if (ok) {
    std::vector<std::string> sources;
    for (int i = 0; i < 4; ++i)
      sources.push_back(emissiveSource(i));
    sources.push_back(brokenSource());
    for (int i = 4; i < 8; ++i)
      sources.push_back(emissiveSource(i));
    auto world = makeWorld(device, sources);
    anari::setAndReleaseParameter(device, frame, "world", world);
    anari::commitParameters(device, frame);

    const double pool = renderPoolMean(device, frame);
    printf("phase3: broken-in-batch pool=%f\n", pool);
    if (!std::isfinite(pool) || pool <= 0.0) {
      fprintf(stderr, "FAIL: phase3 rendered dark after a broken source\n");
      ok = false;
    }
  }

  anari::release(device, frame);
  anari::release(device, device);

  if (!ok)
    return 1;
  printf("mdl concurrent compile passed\n");
  return 0;
}
