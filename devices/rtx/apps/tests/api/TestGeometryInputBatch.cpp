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
#include <dlfcn.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
using vec3 = std::array<float, 3>;
anari::Device device = nullptr;
ANARIArray retireArray = nullptr;
void *sourceA = nullptr;
void *sourceB = nullptr;
const void *trackedA = nullptr;
const void *replacementSource = nullptr;
void *deviceA = nullptr;
bool watchReplacement = false;
bool oldDeviceFreed = false;
bool allocatedBeforeRetirement = false;
cudaStream_t preparationStream = nullptr;
int copies = 0;
int firstSyncCopies = -1;
int preparationSyncs = 0;
int errors = 0;
bool wrongStream = false;
bool retireNext = false;
bool retired = false;

template <typename T>
T next(const char *name)
{
  auto function = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
  if (!function)
    std::abort();
  return function;
}

void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++errors;
    std::fprintf(stderr, "%s\n", message);
  }
}

bool trackedSource(const void *src)
{
  return src && (src == trackedA || src == sourceB || src == replacementSource);
}

void resetCounts()
{
  copies = 0;
  firstSyncCopies = -1;
  preparationSyncs = 0;
  wrongStream = false;
}
} // namespace

extern "C" cudaError_t cudaStreamCreateWithFlags(
    cudaStream_t *stream, unsigned flags)
{
  const auto error = next<decltype(&cudaStreamCreateWithFlags)>(
      "cudaStreamCreateWithFlags")(stream, flags);
  if (error == cudaSuccess && (flags & cudaStreamNonBlocking))
    preparationStream = *stream;
  return error;
}

extern "C" cudaError_t cudaMalloc(void **ptr, size_t bytes)
{
  if (watchReplacement && bytes == sizeof(vec3) && !oldDeviceFreed)
    allocatedBeforeRetirement = true;
  return next<decltype(&cudaMalloc)>("cudaMalloc")(ptr, bytes);
}

extern "C" cudaError_t cudaFree(void *ptr)
{
  const auto error = next<decltype(&cudaFree)>("cudaFree")(ptr);
  if (error == cudaSuccess && watchReplacement && ptr == deviceA)
    oldDeviceFreed = true;
  return error;
}

extern "C" cudaError_t cudaMemcpy(
    void *dst, const void *src, size_t bytes, cudaMemcpyKind kind)
{
  if (kind == cudaMemcpyHostToDevice && trackedSource(src)) {
    ++copies;
    wrongStream = true;
  }
  return next<decltype(&cudaMemcpy)>("cudaMemcpy")(dst, src, bytes, kind);
}

extern "C" cudaError_t cudaMemcpyAsync(void *dst,
    const void *src,
    size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream)
{
  const bool tracked = kind == cudaMemcpyHostToDevice && trackedSource(src);
  if (kind == cudaMemcpyHostToDevice && src == trackedA)
    deviceA = dst;
  if (tracked) {
    ++copies;
    wrongStream |= stream != preparationStream;
  }
  const auto error = next<decltype(&cudaMemcpyAsync)>("cudaMemcpyAsync")(
      dst, src, bytes, kind, stream);
  if (error == cudaSuccess && tracked && src == trackedA && retireNext) {
    retireNext = false;
    // Reenter from the host submission path, not from a CUDA host callback.
    // Public release must wait before the application frees a pinned source.
    const int beforeRelease = preparationSyncs;
    anariRelease(device, retireArray);
    if (preparationSyncs == beforeRelease) {
      ++errors;
      std::fprintf(stderr, "host release omitted pending upload wait\n");
    }
    retireArray = nullptr;
    if (cudaFreeHost(sourceA) != cudaSuccess)
      ++errors;
    sourceA = nullptr;
    retired = true;
  }
  return error;
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
  if (stream == preparationStream)
    ++preparationSyncs;
  if (stream == preparationStream && copies && firstSyncCopies < 0)
    firstSyncCopies = copies;
  return next<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize")(
      stream);
}

int main()
{
  device = makeVisRTXDevice(status);
  if (cudaMallocHost(&sourceA, sizeof(vec3)) != cudaSuccess
      || cudaMallocHost(&sourceB, sizeof(vec3)) != cudaSuccess)
    return 1;
  trackedA = sourceA;
  *static_cast<vec3 *>(sourceA) = {0, 0, 0};
  *static_cast<vec3 *>(sourceB) = {2, 0, 0};
  auto a =
      anariNewArray1D(device, sourceA, nullptr, nullptr, ANARI_FLOAT32_VEC3, 1);
  auto b =
      anariNewArray1D(device, sourceB, nullptr, nullptr, ANARI_FLOAT32_VEC3, 1);
  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::commitParameters(device, material);
  std::array<anari::Geometry, 3> geometries;
  std::array<anari::Surface, 3> surfaces;
  for (size_t i = 0; i < geometries.size(); ++i) {
    auto g = anari::newObject<anari::Geometry>(device, "sphere");
    auto array = i == 1 ? b : a;
    anariSetParameter(device, g, "vertex.position", ANARI_ARRAY1D, &array);
    anari::setParameter(device, g, "radius", 0.25f);
    anari::commitParameters(device, g);
    geometries[i] = g;
    auto s = anari::newObject<anari::Surface>(device);
    anari::setParameter(device, s, "geometry", g);
    anari::setParameter(device, s, "material", material);
    anari::commitParameters(device, s);
    surfaces[i] = s;
  }
  anari::release(device, material);
  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  anari::commitParameters(device, world);
  for (auto surface : surfaces)
    anari::release(device, surface);
  auto bounds = [&](float lo, float hi) {
    std::array<float, 6> result{};
    return anariGetProperty(device,
               world,
               "bounds",
               ANARI_FLOAT32_BOX3,
               result.data(),
               sizeof(result),
               ANARI_WAIT)
        && std::fabs(result[0] - lo) < 1e-4f
        && std::fabs(result[3] - hi) < 1e-4f;
  };
  auto commitGeometryWideAttributes = [&](float value) {
    for (auto geometry : geometries) {
      anari::setParameter(
          device, geometry, "attribute0", std::array<float, 4>{value, 0, 0, 1});
      anari::commitParameters(device, geometry);
    }
  };
  auto change = [&](ANARIArray array, float x) {
    auto *mapped = static_cast<vec3 *>(anariMapArray(device, array));
    if (!mapped)
      std::abort();
    *mapped = {x, 0, 0};
    anariUnmapArray(device, array);
  };
  resetCounts();
  bool ok = bounds(-0.25f, 2.25f) && copies == 2 && firstSyncCopies == 2
      && !wrongStream;
  resetCounts();
  commitGeometryWideAttributes(1.f);
  ok = bounds(-0.25f, 2.25f) && copies == 0 && ok;
  resetCounts();
  change(a, 1.f); // Observer-only fallback still uploads once for shared input.
  ok = bounds(0.75f, 2.25f) && copies == 1 && !wrongStream && ok;
  resetCounts();
  change(a, -1.f);
  change(b, 3.f);
  commitGeometryWideAttributes(2.f);
  ok = bounds(-1.25f, 3.25f) && copies == 2 && firstSyncCopies == 2
      && !wrongStream && ok;
  // Replacement must retire obsolete cached copies before allocating the
  // whole new cohort, rather than keeping both cohorts resident at once.
  void *replacement = nullptr;
  if (cudaMallocHost(&replacement, sizeof(vec3)) != cudaSuccess)
    return 1;
  *static_cast<vec3 *>(replacement) = {-1, 0, 0};
  replacementSource = replacement;
  auto c = anariNewArray1D(
      device, replacement, nullptr, nullptr, ANARI_FLOAT32_VEC3, 1);
  auto replaceShared = [&](ANARIArray array) {
    for (size_t i = 0; i < geometries.size(); ++i) {
      if (i == 1)
        continue;
      anariSetParameter(
          device, geometries[i], "vertex.position", ANARI_ARRAY1D, &array);
      anari::commitParameters(device, geometries[i]);
    }
  };
  watchReplacement = true;
  replaceShared(c);
  ok = bounds(-1.25f, 3.25f) && oldDeviceFreed && !allocatedBeforeRetirement
      && ok;
  watchReplacement = false;
  replaceShared(a);
  ok = bounds(-1.25f, 3.25f) && ok;
  anariRelease(device, c);
  cudaFreeHost(replacement);
  replacementSource = nullptr;
  resetCounts();
  change(a, -2.f);
  commitGeometryWideAttributes(3.f);
  retireArray = a;
  retireNext = true;
  ok = bounds(-2.25f, 3.25f) && retired && copies == 1 && !wrongStream && ok;
  anari::release(device, world);
  for (auto geometry : geometries)
    anari::release(device, geometry);
  if (!retired)
    anariRelease(device, a);
  anariRelease(device, b);
  if (sourceA)
    cudaFreeHost(sourceA);
  cudaFreeHost(sourceB);
  anari::release(device, device);
  ok = errors == 0 && ok;
  std::printf("geometry input batches: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
