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

// MaterialRegistry acquire/release balance: destroying an `mdl` material must
// release its compiled-material registry slot (the source-SWITCH path always
// released; the destructor path historically leaked). Observed through the
// `numRegisteredMdlMaterials` device property; the baseline is captured AFTER
// the first render so it reflects the initialized device, not the pre-init
// null-registry zero. Churning worlds whose `mdl` materials compile DISTINCT
// sources must hold the live-slot count flat at baseline+1 — a destructor
// leak grows it by one per cycle. Two DISTINCT sources alive at once must
// count two slots and dropping one must leave the survivor intact (also the
// guard against a broken-compile fallback collapsing every material onto one
// shared uuid and passing vacuously). Two materials SHARING one source must
// share one slot (refcount, not duplicate), and a world with no MDL-family
// material must return the count to baseline.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
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
static constexpr int CHURN_CYCLES = 3;

// Distinct intensity per cycle => distinct source hash => distinct registry
// slot; the destroyed previous material must give its slot back.
static std::string mdlSource(int cycle)
{
  return std::string(R"mdl(mdl 1.6;
import ::df::*;
export material emissive() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color()mdl")
      + std::to_string(1 + cycle) + R"mdl(.0))));
)mdl";
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

static anari::Surface makeMdlQuad(
    ANARIDevice device, const std::string &source)
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

// One mdl quad per entry in `sources` (empty = matte-only world).
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

static bool expectCount(uint32_t got, uint32_t expected, const char *when)
{
  printf("numRegisteredMdlMaterials %s = %u (expected %u)\n",
      when,
      got,
      expected);
  if (got != expected) {
    fprintf(stderr,
        "FAIL: %s: expected %u registered material(s), got %u — registry "
        "slot leak?\n",
        when,
        expected,
        got);
    return false;
  }
  return true;
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
  anari::setParameter(device, renderer, "pixelSamples", 1);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", IMAGE_SIZE);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);

  auto renderWorld = [&](anari::World world) {
    anari::setAndReleaseParameter(device, frame, "world", world);
    anari::commitParameters(device, frame);
    anari::render(device, frame);
    anari::wait(device, frame);
  };

  bool ok = true;

  // Baseline AFTER the first (mdl-free) render: the device — and under lazy
  // init the MDL subsystem — is up, so any init-time registrations are
  // absorbed instead of skewing every later assertion.
  renderWorld(makeWorld(device, {}));
  const uint32_t baseline = queryCount(device);

  // Churn: each cycle swaps in a DISTINCT source; the previous cycle's
  // destroyed material must release its slot, holding the count flat.
  for (int cycle = 0; cycle < CHURN_CYCLES; ++cycle) {
    renderWorld(makeWorld(device, {mdlSource(cycle)}));
    char when[64];
    snprintf(when, sizeof(when), "after churn cycle %d", cycle);
    ok = expectCount(queryCount(device), baseline + 1, when) && ok;
  }

  // Two DISTINCT sources alive at once: two slots. This also guards the churn
  // phase against passing vacuously — a broken compile collapses every
  // material onto the shared fallback uuid, which reads one slot, not two.
  renderWorld(makeWorld(device, {mdlSource(0), mdlSource(1)}));
  ok = expectCount(
           queryCount(device), baseline + 2, "with two distinct sources")
      && ok;

  // Drop one of the two: the survivor's slot (and its SBT index) must ride
  // through the other slot's erasure.
  renderWorld(makeWorld(device, {mdlSource(1)}));
  ok = expectCount(
           queryCount(device), baseline + 1, "after dropping one of two")
      && ok;

  // Shared source: two materials, one compiled slot.
  renderWorld(makeWorld(device, {mdlSource(0), mdlSource(0)}));
  ok = expectCount(queryCount(device), baseline + 1, "with shared source x2")
      && ok;

  // No MDL-family material left: back to baseline.
  renderWorld(makeWorld(device, {}));
  ok = expectCount(queryCount(device), baseline, "after mdl-free world") && ok;

  anari::release(device, frame);
  anari::release(device, device);

  if (!ok)
    return 1;
  printf("mdl material registry release passed\n");
  return 0;
}
