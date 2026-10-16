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

// ELF interposition injects recoverable CUDA API errors without poisoning the
// real CUDA context. This test requires a shared CUDA runtime, not production
// failure-injection switches in the device.
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
enum class Failure
{
  NONE,
  ALLOCATION,
  COPY,
  WORK_SYNC,
  COPY_SYNC
};
Failure failure = Failure::NONE;
size_t allocationBytes = 0;
int injections = 0;
int errors = 0;
std::atomic<int> completions{0};

bool inject(Failure kind)
{
  if (failure != kind)
    return false;
  failure = Failure::NONE;
  ++injections;
  return true;
}

template <typename T>
T next(const char *name)
{
  auto result = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
  if (!result) {
    fprintf(stderr, "missing CUDA symbol %s\n", name);
    std::abort();
  }
  return result;
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
    fprintf(stderr, "%s\n", message);
  }
}
void complete(const void *, ANARIDevice, ANARIFrame)
{
  ++completions;
}
} // namespace

extern "C" cudaError_t cudaMalloc(void **ptr, size_t bytes)
{
  if (bytes == allocationBytes && inject(Failure::ALLOCATION)) {
    // Use the real runtime: returning an enum alone does not set CUDA
    // last-error state, and misses contamination of the next preparation.
    *ptr = nullptr;
    return next<decltype(&cudaMalloc)>("cudaMalloc")(ptr, size_t(-1));
  }
  return next<decltype(&cudaMalloc)>("cudaMalloc")(ptr, bytes);
}

extern "C" cudaError_t cudaMemcpy(
    void *dst, const void *src, size_t bytes, cudaMemcpyKind kind)
{
  if (kind == cudaMemcpyDeviceToDevice && inject(Failure::COPY))
    return cudaErrorInvalidValue;
  return next<decltype(&cudaMemcpy)>("cudaMemcpy")(dst, src, bytes, kind);
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
  // Drain real work even when simulating a failed completion report.
  const auto result =
      next<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize")(stream);
  if ((stream && inject(Failure::WORK_SYNC))
      || (!stream && inject(Failure::COPY_SYNC)))
    return cudaErrorUnknown;
  return result;
}

int main(int argc, char **argv)
{
  const bool isosurfaceFailure =
      argc > 1 && std::strcmp(argv[1], "isosurface") == 0;
  const bool sphereFailure = argc > 1
      && (std::strcmp(argv[1], "sphere") == 0
          || std::strcmp(argv[1], "sphere-indexed") == 0);
  const bool curveFailure = argc == 1 || isosurfaceFailure || sphereFailure;
  auto d = makeVisRTXDevice(status);
  using vec3 = std::array<float, 3>;
  const std::array<vec3, 6> vertices = {
      {{-1, 0, 0}, {1, 0, 0}, {-1, 1, 0}, {1, 1, 0}, {-1, 2, 0}, {1, 2, 0}}};
  void *deviceVertices = nullptr;
  if (cudaMalloc(&deviceVertices, sizeof(vertices)) != cudaSuccess
      || cudaMemcpy(deviceVertices,
             vertices.data(),
             sizeof(vertices),
             cudaMemcpyHostToDevice)
          != cudaSuccess)
    return 1;
  auto array = anariNewArray1D(
      d, deviceVertices, nullptr, nullptr, ANARI_FLOAT32_VEC3, vertices.size());
  auto geometry = anari::newObject<anari::Geometry>(d,
      isosurfaceFailure   ? "isosurface"
          : sphereFailure ? "sphere"
                          : "curve");
  if (sphereFailure && std::strcmp(argv[1], "sphere-indexed") == 0) {
    const std::array<unsigned, 3> indices{5, 1, 3};
    anari::setParameterArray1D(
        d, geometry, "primitive.index", indices.data(), indices.size());
  }
  if (isosurfaceFailure) {
    std::array<float, 512> values{};
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = float(i / 64);
    auto field = anari::newObject<anari::SpatialField>(d, "structuredRegular");
    auto data = anari::newArray3D(d, values.data(), 8, 8, 8);
    anari::setAndReleaseParameter(d, field, "data", data);
    anari::setParameter(d, field, "origin", vec3{0, 0, 0});
    anari::setParameter(d, field, "spacing", vec3{1, 1, 1});
    anari::commitParameters(d, field);
    anari::setAndReleaseParameter(d, geometry, "field", field);
    anari::setParameter(d, geometry, "isovalue", 3.5f);
  } else {
    anari::setParameter(d, geometry, "vertex.position", array);
    anari::setParameter(d, geometry, "radius", 0.1f);
  }
  anari::commitParameters(d, geometry);
  auto material = anari::newObject<anari::Material>(d, "matte");
  anari::commitParameters(d, material);
  auto surface = anari::newObject<anari::Surface>(d);
  anari::setParameter(d, surface, "geometry", geometry);
  anari::setAndReleaseParameter(d, surface, "material", material);
  anari::commitParameters(d, surface);
  std::vector<anari::Surface> surfaces{surface};
  anari::Geometry sibling = nullptr;
  if (curveFailure && !isosurfaceFailure && !sphereFailure) {
    sibling = anari::newObject<anari::Geometry>(d, "curve");
    anari::setParameterArray1D(
        d, sibling, "vertex.position", vertices.data(), 4);
    anari::setParameter(d, sibling, "radius", 0.1f);
    anari::commitParameters(d, sibling);
    auto siblingMaterial = anari::newObject<anari::Material>(d, "matte");
    anari::commitParameters(d, siblingMaterial);
    auto siblingSurface = anari::newObject<anari::Surface>(d);
    anari::setParameter(d, siblingSurface, "geometry", sibling);
    anari::setAndReleaseParameter(
        d, siblingSurface, "material", siblingMaterial);
    anari::commitParameters(d, siblingSurface);
    surfaces.push_back(siblingSurface);
  }
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(
      d, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(d, s);
  anari::commitParameters(d, world);
  auto camera = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, camera, "position", vec3{0, 0, -3});
  anari::setParameter(d, camera, "direction", vec3{0, 0, 1});
  anari::commitParameters(d, camera);
  auto renderer = anari::newObject<anari::Renderer>(d, "debug");
  anari::commitParameters(d, renderer);
  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", std::array<unsigned, 2>{8, 8});
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(d, frame, "camera", camera);
  anari::setAndReleaseParameter(d, frame, "renderer", renderer);
  anari::setParameter(d, frame, "world", world);
  ANARIFrameCompletionCallback callback = complete;
  anari::setParameter(d,
      frame,
      "frameCompletionCallback",
      ANARI_FRAME_COMPLETION_CALLBACK,
      &callback);
  anari::commitParameters(d, frame);
  std::array<float, 6> bounds{};
  auto haveBounds = [&]() {
    return anariGetProperty(d,
               world,
               "bounds",
               ANARI_FLOAT32_BOX3,
               bounds.data(),
               sizeof(bounds),
               ANARI_WAIT)
        != 0;
  };
  bool ok = errors == 0;
  if (curveFailure) {
    if (isosurfaceFailure) {
      ok = haveBounds() && ok;
      anari::setParameter(d, geometry, "isovalue", 2.5f);
      anari::commitParameters(d, geometry);
    }
    if (sibling) {
      anari::setParameter(d, sibling, "radius", 0.15f);
      anari::commitParameters(d, sibling);
    }
    // Sphere AABBs (two vec3s each), or generated curve indices.
    allocationBytes = sphereFailure
        ? (std::strcmp(argv[1], "sphere-indexed") == 0 ? 3 : vertices.size())
            * 2 * sizeof(vec3)
        : 3 * sizeof(unsigned);
    failure = isosurfaceFailure ? Failure::WORK_SYNC : Failure::ALLOCATION;
    ok = !haveBounds() && ok;
    // The unaffected sibling must not inherit the first geometry's OOM.
    // Recovery below must succeed without recommitting the sibling.
    ok = injections == 1 && errors == 1 && ok;
    anari::render(d, frame);
    anari::wait(d, frame);
    ok = completions == 0 && ok;
    const int beforeErrors = errors;
    anari::setParameter(
        d, geometry, isosurfaceFailure ? "isovalue" : "radius", 0.2f);
    anari::commitParameters(d, geometry);
    ok = haveBounds() && errors == beforeErrors && ok;
    anari::render(d, frame);
    anari::wait(d, frame);
    ok = completions == 1 && errors == beforeErrors && ok;
    anari::release(d, array);
  } else {
    ok = haveBounds() && ok;
    anari::render(d, frame);
    anari::wait(d, frame);
    ok = completions == 1 && errors == 0 && ok;
    allocationBytes = sizeof(vertices);
    if (std::strcmp(argv[1], "allocation") == 0)
      failure = Failure::ALLOCATION;
    else if (std::strcmp(argv[1], "copy") == 0)
      failure = Failure::COPY;
    else if (std::strcmp(argv[1], "work-sync") == 0)
      failure = Failure::WORK_SYNC;
    else if (std::strcmp(argv[1], "copy-sync") == 0)
      failure = Failure::COPY_SYNC;
    else
      return 2;
    anari::release(d, array);
    ok = injections == 1 && errors > 0 && ok;
    cudaFree(deviceVertices);
    deviceVertices = nullptr;
    ok = !haveBounds() && ok;
    anari::render(d, frame);
    anari::wait(d, frame);
    ok = completions == 1 && ok;
    // No retry or device re-enable after the injected CUDA error is gone.
    anari::setParameter(d, geometry, "radius", 0.3f);
    anari::commitParameters(d, geometry);
    ok = !haveBounds() && ok;
  }
  if (deviceVertices)
    cudaFree(deviceVertices);
  anari::release(d, frame);
  anari::release(d, world);
  anari::release(d, geometry);
  if (sibling)
    anari::release(d, sibling);
  anari::release(d, d);
  printf("%s injections=%d %s\n",
      curveFailure ? "curve" : argv[1],
      injections,
      ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
