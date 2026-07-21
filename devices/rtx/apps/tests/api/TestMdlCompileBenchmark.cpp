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

// Wall-clock of compiling a batch of distinct `mdl` materials in one flush.
// It gates correctness (every material registers) and prints the elapsed time
// and the active VISRTX_MDL_COMPILE_THREADS so the compile speedup of ADR 0009
// can be read off by running it at different thread counts. Not a timing gate:
// absolute times are machine-dependent, so it never fails on duration.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using vec3 = std::array<float, 3>;

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

static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;
static constexpr int MATERIALS = 64;

// A body of arithmetic makes each compile non-trivial so the compile stage, not
// fixed overhead, dominates the measured time. The terms are summed (linear
// growth) -- nesting would blow the source up exponentially.
static std::string heavySource(int index)
{
  std::string expr = std::to_string(1 + index) + ".0";
  for (int i = 1; i <= 48; ++i)
    expr += " + math::sin(" + std::to_string(index + i) + ".0) * math::cos("
        + std::to_string(index * i + 1) + ".0)";
  return "mdl 1.6;\nimport ::df::*;\nimport ::math::*;\n"
         "export material emissive() = material(\n"
         "    surface: material_surface(\n"
         "        emission: material_emission(\n"
         "            emission: df::diffuse_edf(),\n"
         "            intensity: color("
      + expr + "))));\n";
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

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  std::vector<anari::Surface> surfaces;
  for (int i = 0; i < MATERIALS; ++i)
    surfaces.push_back(makeMdlQuad(device, heavySource(i)));

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(device, s);

  // Commit the world and force the flush that compiles the whole batch, timing
  // exactly that.
  const auto start = std::chrono::steady_clock::now();
  anari::commitParameters(device, world);
  uint32_t count = 0;
  anari::getProperty(
      device, device, "numRegisteredMdlMaterials", count, ANARI_WAIT);
  const auto end = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  const char *threads = std::getenv("VISRTX_MDL_COMPILE_THREADS");
  printf("compiled %u/%d materials in %.1f ms (VISRTX_MDL_COMPILE_THREADS=%s)\n",
      count, MATERIALS, ms, threads ? threads : "default");

  anari::release(device, world);
  anari::release(device, device);

  if (count != uint32_t(MATERIALS)) {
    fprintf(stderr, "FAIL: expected %d registered materials, got %u\n",
        MATERIALS, count);
    return 1;
  }
  printf("mdl compile benchmark passed\n");
  return 0;
}
