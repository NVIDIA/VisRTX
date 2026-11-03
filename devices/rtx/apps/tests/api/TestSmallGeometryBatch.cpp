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
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "DeviceTestArray.h"
#ifdef VISRTX_TEST_INTERPOSE_CUDA
#include <dlfcn.h>
#endif

namespace {
#ifdef VISRTX_TEST_INTERPOSE_CUDA
bool trackSummaries = false;
int summaryCopies = 0;
#endif
using vec3 = std::array<float, 3>;
int errors = 0;
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
struct Case
{
  std::string type;
  size_t count;
  anari::Geometry geometry;
  anari::Group group;
  std::array<float, 6> bounds;
};

bool run(bool device)
{
  auto d = makeVisRTXDevice(status);
  auto material = anari::newObject<anari::Material>(d, "physicallyBased");
  anari::setParameter(d, material, "emissive", vec3{1, 1, 1});
  anari::commitParameters(d, material);
  std::vector<Case> cases;
  std::vector<anari::Instance> instances;
  for (const char *type : {"sphere", "cylinder", "cone", "triangle"})
    for (size_t n : {1, 31, 32, 33, 255, 256, 257}) {
      auto g = anari::newObject<anari::Geometry>(d, type);
      auto surface = anari::newObject<anari::Surface>(d);
      anari::setParameter(d, surface, "geometry", g);
      anari::setParameter(d, surface, "material", material);
      anari::commitParameters(d, surface);
      auto group = anari::newObject<anari::Group>(d);
      anari::setParameterArray1D(d, group, "surface", &surface, 1);
      anari::release(d, surface);
      anari::commitParameters(d, group);
      auto instance = anari::newObject<anari::Instance>(d, "transform");
      anari::setParameter(d, instance, "group", group);
      anari::commitParameters(d, instance);
      instances.push_back(instance);
      cases.push_back({type, n, g, group, {}});
    }
  anari::release(d, material);
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(
      d, world, "instance", instances.data(), instances.size());
  anari::commitParameters(d, world);
  for (auto instance : instances)
    anari::release(d, instance);
  bool ok = true;
  for (int generation = 0; generation < 3; ++generation) {
    for (auto &c : cases) {
      // Reuse a batch's slots across degenerate results, then swap small and
      // large paths to catch stale result pointers and scratch-growth errors.
      const size_t n = generation == 2 ? (c.count <= 256 ? 257 : 32) : c.count;
      const float r = generation == 1 ? 0.f : 0.25f;
      const float h = generation == 1 ? 0.f : 0.5f;
      std::vector<vec3> vertices;
      std::vector<float> radii;
      for (size_t i = 0; i < n; ++i) {
        const float x = 2.f * i;
        if (c.type == "sphere")
          vertices.push_back({x, 0, 0});
        else if (c.type == "triangle") {
          vertices.push_back({x - r, -h, 0});
          vertices.push_back({x + r, -h, 0});
          vertices.push_back({x, h, 0});
        } else {
          vertices.push_back({x, -h, 0});
          vertices.push_back({x, h, 0});
          radii.push_back(r * 0.5f);
          radii.push_back(r);
        }
      }
      setTestArray(d, c.geometry, "vertex.position", vertices, device, 1);
      if (c.type == "cone")
        setTestArray(d, c.geometry, "vertex.radius", radii, device, 1);
      anari::setParameter(d, c.geometry, "radius", r);
      anari::setParameter(d, c.geometry, "caps", "both");
      anari::commitParameters(d, c.geometry);
      const float y = c.type == "sphere" ? r : c.type == "triangle" ? h : h + r;
      const float z = c.type == "triangle" ? 0.f : r;
      c.bounds = {-r, -y, -z, 2.f * (n - 1) + r, y, z};
    }
    // One world query flushes all geometries together, including lazy area
    // requests on the first generation and bounds+area jobs on later ones.
    std::array<float, 6> worldBounds{};
#ifdef VISRTX_TEST_INTERPOSE_CUDA
    summaryCopies = 0;
    trackSummaries = true;
#endif
    ok = anariGetProperty(d,
             world,
             "bounds",
             ANARI_FLOAT32_BOX3,
             worldBounds.data(),
             sizeof(worldBounds),
             ANARI_WAIT)
        && ok;
#ifdef VISRTX_TEST_INTERPOSE_CUDA
    trackSummaries = false;
    // Initial bounds and lazy areas are separate collections. Once area data
    // is wanted, finalization batches both together. Large-path scalar copies
    // and OptiX's 8/24-byte readbacks are not batch summaries.
    const int expectedCopies = generation == 0 ? 2 : 1;
    if (summaryCopies != expectedCopies)
      std::fprintf(stderr,
          "generation=%d summary copies=%d expected=%d\n",
          generation,
          summaryCopies,
          expectedCopies);
    ok = summaryCopies == expectedCopies && ok;
#endif
    uint32_t lights = 0;
    ok = anariGetProperty(d,
             world,
             "numLightInstances",
             ANARI_UINT32,
             &lights,
             sizeof(lights),
             ANARI_WAIT)
        && lights == (generation == 1 ? 0 : cases.size()) && ok;
    for (const auto &c : cases) {
      std::array<float, 6> actual{};
      bool valid = anariGetProperty(d,
          c.group,
          "bounds",
          ANARI_FLOAT32_BOX3,
          actual.data(),
          sizeof(actual),
          ANARI_WAIT);
      if (generation == 1) {
        // OptiX drops zero-extent primitives from its emitted BVH bounds.
        for (int axis = 0; axis < 3; ++axis)
          valid = actual[axis] > actual[axis + 3] && valid;
      } else {
        // OptiX conservatively pads bounds in proportion to coordinate scale.
        const float tolerance = 1e-4f + 1e-6f * c.bounds[3];
        for (int i = 0; i < 6; ++i)
          valid = std::isfinite(actual[i])
              && std::abs(actual[i] - c.bounds[i]) < tolerance && valid;
      }
      if (!valid)
        std::fprintf(stderr,
            "%s count=%zu generation=%d bounds actual=(%g,%g,%g;%g,%g,%g) expected=(%g,%g,%g;%g,%g,%g)\n",
            c.type.c_str(),
            c.count,
            generation,
            actual[0],
            actual[1],
            actual[2],
            actual[3],
            actual[4],
            actual[5],
            c.bounds[0],
            c.bounds[1],
            c.bounds[2],
            c.bounds[3],
            c.bounds[4],
            c.bounds[5]);
      ok = valid && ok;
    }
    ok = errors == 0 && ok;
  }
  anari::release(d, world);
  for (const auto &c : cases) {
    anari::release(d, c.group);
    anari::release(d, c.geometry);
  }
  anari::release(d, d);
  std::printf(
      "small geometry batch device=%d %s\n", device, ok ? "PASS" : "FAIL");
  return ok;
}
} // namespace

#ifdef VISRTX_TEST_INTERPOSE_CUDA
extern "C" cudaError_t cudaMemcpyAsync(void *dst,
    const void *src,
    size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream)
{
  constexpr size_t pairBytes = 2 * (sizeof(float) + sizeof(uint32_t));
  if (trackSummaries && kind == cudaMemcpyDeviceToHost && bytes
      && bytes % pairBytes == 0)
    ++summaryCopies;
  using Copy = cudaError_t (*)(
      void *, const void *, size_t, cudaMemcpyKind, cudaStream_t);
  auto real = reinterpret_cast<Copy>(dlsym(RTLD_NEXT, "cudaMemcpyAsync"));
  if (!real)
    std::abort();
  return real(dst, src, bytes, kind, stream);
}
#endif

int main()
{
  bool ok = run(false);
  ok = run(true) && ok;
  return ok ? 0 : 1;
}
