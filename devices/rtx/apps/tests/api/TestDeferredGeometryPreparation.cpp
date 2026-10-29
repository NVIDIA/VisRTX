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

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <cuda_runtime_api.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

namespace {
int freed = 0;
int errors = 0;
int rejectedQueries = 0;
anari::World callbackWorld = nullptr;

void releaseDevice(const void *, const void *memory)
{
  cudaFree(const_cast<void *>(memory));
  ++freed;
}

void status(const void *,
    ANARIDevice d,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity > ANARI_SEVERITY_ERROR)
    return;
  ++errors;
  std::fprintf(stderr, "%s\n", message);
  if (callbackWorld) {
    std::array<float, 6> bounds{};
    if (!anariGetProperty(d,
            callbackWorld,
            "bounds",
            ANARI_FLOAT32_BOX3,
            bounds.data(),
            sizeof(bounds),
            ANARI_WAIT))
      ++rejectedQueries;
  }
}

bool position(anari::Device d, anari::Geometry geometry, float x)
{
  const std::array<float, 3> center{x, 0, 0};
  void *device = nullptr;
  if (cudaMalloc(&device, sizeof(center)) != cudaSuccess
      || cudaMemcpy(
             device, center.data(), sizeof(center), cudaMemcpyHostToDevice)
          != cudaSuccess)
    return false;
  auto array =
      anariNewArray1D(d, device, releaseDevice, nullptr, ANARI_FLOAT32_VEC3, 1);
  anariSetParameter(d, geometry, "vertex.position", ANARI_ARRAY1D, &array);
  anariRelease(d, array);
  anari::setParameter(d, geometry, "radius", 1.f);
  anari::commitParameters(d, geometry);
  return true;
}

bool flushGeometry(anari::Device d, anari::Geometry geometry)
{
  uint8_t valid = 0;
  return anariGetProperty(d,
             geometry,
             "valid",
             ANARI_BOOL,
             &valid,
             sizeof(valid),
             ANARI_WAIT)
      && valid;
}
} // namespace

int main()
{
  auto d = makeVisRTXDevice(status);
  auto geometry = anari::newObject<anari::Geometry>(d, "sphere");
  if (!position(d, geometry, 0.f))
    return 1;
  auto material = anari::newObject<anari::Material>(d, "matte");
  anari::commitParameters(d, material);
  auto surface = anari::newObject<anari::Surface>(d);
  anari::setParameter(d, surface, "geometry", geometry);
  anari::setAndReleaseParameter(d, surface, "material", material);
  anari::commitParameters(d, surface);
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, world, "surface", &surface, 1);
  anari::release(d, surface);
  anari::commitParameters(d, world);

  // Flush finalizers without requesting a BVH: readbacks remain queued.
  bool ok = flushGeometry(d, geometry) && freed == 0;
  ok = position(d, geometry, 4.f) && ok;
  ok = flushGeometry(d, geometry) && freed == 1 && ok;
  std::array<float, 6> bounds{};
  auto haveBounds = [&] {
    return anariGetProperty(d,
               world,
               "bounds",
               ANARI_FLOAT32_BOX3,
               bounds.data(),
               sizeof(bounds),
               ANARI_WAIT)
        != 0;
  };
  ok = haveBounds() && std::fabs(bounds[0] - 3.f) < 1e-4f
      && std::fabs(bounds[3] - 5.f) < 1e-4f && errors == 0 && ok;

  // A failed, unreferenced geometry must not poison an unrelated world. A
  // status callback must not reenter collection and publish partial results.
  auto invalid = anari::newObject<anari::Geometry>(d, "sphere");
  ok = position(d, invalid, std::numeric_limits<float>::quiet_NaN()) && ok;
  ok = flushGeometry(d, invalid) && ok;
  callbackWorld = world;
  ok = haveBounds() && errors == 1 && rejectedQueries == 1 && ok;
  callbackWorld = nullptr;
  anari::release(d, invalid);

  // The pending queue owns a last-reference geometry until device teardown.
  auto orphan = anari::newObject<anari::Geometry>(d, "sphere");
  ok = position(d, orphan, 8.f) && flushGeometry(d, orphan) && ok;
  anari::release(d, orphan);
  anari::release(d, world);
  anari::release(d, geometry);
  ok = freed == 3 && ok;
  anari::release(d, d);
  ok = freed == 4 && errors == 1 && ok;
  std::printf("deferred geometry: freed=%d errors=%d %s\n",
      freed,
      errors,
      ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
