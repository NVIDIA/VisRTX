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

// The `mdlPtxFingerprint` device property is a content hash of every compiled
// MDL material's PTX. It exists to catch a silent miscompile from the parallel
// compile path (ADR 0009): PTX corruption that never changes a rendered image
// still changes the fingerprint. The CTest wrapper runs this binary under
// VISRTX_MDL_COMPILE_THREADS=1 and =8 and asserts the printed fingerprints
// match; this binary also requires all requested materials to compile without
// errors, so identical fallback materials cannot pass the gate.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;

static std::atomic<unsigned> errorCount{0};

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
  } else if (severity == ANARI_SEVERITY_ERROR) {
    errorCount.fetch_add(1, std::memory_order_relaxed);
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
  }
}

static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;
static constexpr int DISTINCT_MATERIALS = 24;

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

static uint64_t queryFingerprint(ANARIDevice device)
{
  uint64_t fp = 0;
  if (!anari::getProperty(
          device, device, "mdlPtxFingerprint", fp, ANARI_WAIT)) {
    fprintf(stderr, "FAIL: device has no mdlPtxFingerprint property\n");
    std::exit(1);
  }
  return fp;
}

int main(int argc, char **argv)
{
  // Negative CTests prove both total and partial fallback fail the gate. With
  // one failed material the slot count can still be 24 (23 + default), so the
  // error callback is independently necessary.
  const std::string mode = argc == 2 ? argv[1] : "";
  auto device = makeVisRTXDevice(statusFunc);

  std::vector<anari::Surface> surfaces;
  for (int i = 0; i < DISTINCT_MATERIALS; ++i) {
    const bool broken = mode == "--invalid-source"
        || (mode == "--one-invalid-source" && i == 0);
    surfaces.push_back(makeMdlQuad(device,
        broken ? "mdl 1.6; export material emissive() = invalid;"
               : emissiveSource(i)));
  }

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(device, s);
  anari::commitParameters(device, world);

  const uint64_t fp1 = queryFingerprint(device);
  const uint64_t fp2 = queryFingerprint(device);

  uint32_t count = 0;
  bool ok = anari::getProperty(
      device, device, "numRegisteredMdlMaterials", count, ANARI_WAIT);
  if (!ok || count != DISTINCT_MATERIALS) {
    fprintf(stderr,
        "FAIL: expected %d compiled materials, got %u\n",
        DISTINCT_MATERIALS,
        count);
    ok = false;
  }
  if (fp1 == 0) {
    fprintf(stderr,
        "FAIL: fingerprint is zero for %d compiled materials\n",
        DISTINCT_MATERIALS);
    ok = false;
  }
  if (fp1 != fp2) {
    fprintf(stderr,
        "FAIL: fingerprint not stable across queries (%016llx vs %016llx)\n",
        (unsigned long long)fp1,
        (unsigned long long)fp2);
    ok = false;
  }

  anari::release(device, world);
  anari::release(device, device);

  const auto errors = errorCount.load(std::memory_order_relaxed);
  if (errors != 0) {
    fprintf(stderr, "FAIL: %u unexpected ANARI errors\n", errors);
    ok = false;
  }
  if (!ok)
    return 1;
  // Only publish a fingerprint after validating the workload and callbacks.
  printf("mdlPtxFingerprint=%016llx\n", (unsigned long long)fp1);
  printf("mdl ptx fingerprint passed\n");
  return 0;
}
