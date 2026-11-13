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
#include <atomic>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>
#include "DeviceTestArray.h"
#include "geometry/Geometry.h"

namespace {
using vec3 = std::array<float, 3>;
using uvec2 = std::array<uint32_t, 2>;
using uvec3 = std::array<uint32_t, 3>;
int errors = 0;
std::atomic<int> completions{0};
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
void complete(const void *, ANARIDevice, ANARIFrame)
{
  ++completions;
}

struct Case
{
  std::string type;
  size_t count;
  bool indexed;
  bool device;
  anari::Geometry geometry;
  std::vector<double> areas;
};

float weight(size_t i, int generation)
{
  if (generation == 1)
    return 0.f;
  // Put zeros on both sides of warp and small/general boundaries, then move
  // them on the recovery generation. Sharp positive weights also cross warps.
  const bool zero = generation == 0 ? (i == 0 || i == 30 || i == 31 || i == 32
                                          || i == 254 || i == 255 || i == 256)
                                    : (i % 13 == 0);
  if (zero)
    return 0.f;
  const float value = float((17 * i + 3 + generation) % 23 + 1) / 8.f;
  return (i == 29 || i == 33 || i == 253 || i == 255) ? 16.f * value : value;
}

void author(anari::Device d, Case &c, int generation)
{
  std::vector<vec3> vertices;
  std::vector<float> radii;
  std::vector<uint8_t> caps;
  std::vector<uint32_t> sphereIndices;
  std::vector<uvec2> segmentIndices;
  std::vector<uvec3> triangleIndices;
  const auto n = c.count;
  for (size_t i = 0; i < n; ++i) {
    const float x = 100.f * i;
    const float w = weight(i, generation);
    const float length = float(i % 5 + 1) / 2.f;
    if (c.type == "triangle") {
      // Independent closed form: base 2*w, height 1, hence area w.
      vertices.push_back({x, 0, 0});
      vertices.push_back({x + 2.f * w, 0, 0});
      vertices.push_back({x, 1, 0});
    } else if (c.type == "sphere") {
      vertices.push_back({x, 0, 0});
      radii.push_back(w);
    } else {
      vertices.push_back({x, 0, 0});
      vertices.push_back({x, length, 0});
      caps.push_back(i % 3 == 0);
      caps.push_back(i % 5 == 0);
      if (c.type == "cone") {
        radii.push_back(w / 2.f);
        radii.push_back(w);
      } else
        radii.push_back(w); // Cylinder radius is per primitive, not per vertex.
    }
    const uint32_t source = uint32_t(c.indexed ? n - 1 - i : i);
    sphereIndices.push_back(source);
    segmentIndices.push_back({2 * source, 2 * source + 1});
    triangleIndices.push_back({3 * source, 3 * source + 1, 3 * source + 2});
  }
  // Use independent analytic areas in double precision, not the production
  // primitiveArea helper, scan implementation or published total as an oracle.
  const double pi = std::acos(-1.0);
  c.areas.clear();
  for (size_t i = 0; i < n; ++i) {
    const size_t source = c.indexed ? n - 1 - i : i;
    const double w = weight(source, generation);
    const double length = double(source % 5 + 1) / 2.0;
    double area = 0;
    if (c.type == "triangle")
      area = w;
    else if (c.type == "sphere")
      area = 4.0 * pi * w * w;
    else {
      const double r0 = c.type == "cone" ? w / 2.0 : weight(i, generation);
      const double r1 = c.type == "cone" ? w : r0;
      area = c.type == "cone"
          ? pi * (r0 + r1) * std::sqrt(length * length + (r0 - r1) * (r0 - r1))
          : 2.0 * pi * r0 * length;
      if (source % 3 == 0)
        area += pi * r0 * r0;
      if (source % 5 == 0)
        area += pi * r1 * r1;
    }
    c.areas.push_back(area);
  }
  setTestArray(d, c.geometry, "vertex.position", vertices, c.device, 1);
  if (c.type != "triangle")
    setTestArray(d,
        c.geometry,
        c.type == "cylinder" ? "primitive.radius" : "vertex.radius",
        radii,
        c.device,
        1);
  if (!caps.empty())
    setTestArray(d, c.geometry, "vertex.cap", caps, c.device, 1);
  if (c.indexed) {
    if (c.type == "triangle")
      setTestArray(
          d, c.geometry, "primitive.index", triangleIndices, c.device, 1);
    else if (c.type == "sphere")
      setTestArray(
          d, c.geometry, "primitive.index", sphereIndices, c.device, 1);
    else
      setTestArray(
          d, c.geometry, "primitive.index", segmentIndices, c.device, 1);
  }
  anari::commitParameters(d, c.geometry);
}

bool checkPublished(const Case &c, int generation)
{
  // Native VisRTX handles are Geometry pointers. Inspect the actual published
  // GPU record only after a completed frame. This internal test is compiled
  // with the device's layout definitions (not a new public ANARI property).
  auto *geometry = reinterpret_cast<visrtx::Geometry *>(c.geometry);
  auto *state = static_cast<visrtx::DeviceGlobalState *>(
      geometry->helium::BaseObject::deviceState());
  const auto *registry = state->registry.geometries.devicePtr();
  visrtx::GeometryGPUData data{};
  if (!registry
      || cudaMemcpy(&data,
             registry + geometry->index(),
             sizeof(data),
             cudaMemcpyDeviceToHost)
          != cudaSuccess)
    return false;
  const float *cdf = nullptr;
  uint32_t count = 0;
  float total = -1;
  auto sampling = [&](const auto &record) {
    cdf = record.primAreaCdf;
    count = record.numPrimitives;
    total = record.totalArea;
  };
  if (c.type == "triangle" && data.type == visrtx::GeometryType::TRIANGLE)
    sampling(data.tri);
  else if (c.type == "sphere" && data.type == visrtx::GeometryType::SPHERE)
    sampling(data.sphere);
  else if (c.type == "cone" && data.type == visrtx::GeometryType::CONE)
    sampling(data.cone);
  else if (c.type == "cylinder" && data.type == visrtx::GeometryType::CYLINDER)
    sampling(data.cylinder);
  else
    return false;
  const double expectedTotal =
      std::accumulate(c.areas.begin(), c.areas.end(), 0.0);
  bool ok = std::isfinite(total);
  if (expectedTotal == 0)
    ok = total == 0.f && count == 0 && !cdf && ok;
  else {
    ok = cdf && count == c.count
        && std::abs(double(total) - expectedTotal) <= 2e-6 * expectedTotal
        && ok;
    if (!cdf || count != c.count)
      return false;
    std::vector<float> values(count);
    if (cudaMemcpy(
            values.data(), cdf, count * sizeof(float), cudaMemcpyDeviceToHost)
        != cudaSuccess)
      return false;
    double prefix = 0;
    float previous = 0;
    for (size_t i = 0; i < count; ++i) {
      prefix += c.areas[i];
      const double expected = prefix / expectedTotal;
      const bool entryOk = std::isfinite(values[i]) && values[i] >= previous
          && values[i] >= 0.f && values[i] <= 1.f
          && std::abs(double(values[i]) - expected) <= 2e-6
          && (c.areas[i] != 0 || std::abs(values[i] - previous) <= 2e-6f);
      if (!entryOk) {
        std::fprintf(stderr,
            "%s n=%zu indexed=%d device=%d gen=%d bin=%zu actual=%.9g expected=%.9g\n",
            c.type.c_str(),
            c.count,
            c.indexed,
            c.device,
            generation,
            i,
            values[i],
            expected);
        ok = false;
        break;
      }
      previous = values[i];
    }
    ok = values.back() == 1.f && ok;
  }
  if (!ok)
    std::fprintf(stderr,
        "%s n=%zu indexed=%d device=%d gen=%d total=%.9g expected=%.9g\n",
        c.type.c_str(),
        c.count,
        c.indexed,
        c.device,
        generation,
        total,
        expectedTotal);
  return ok;
}
} // namespace

int main()
{
  auto d = makeVisRTXDevice(status);
  auto material = anari::newObject<anari::Material>(d, "physicallyBased");
  anari::setParameter(d, material, "emissive", vec3{1, 1, 1});
  anari::commitParameters(d, material);
  std::vector<Case> cases;
  std::vector<anari::Surface> surfaces;
  constexpr size_t boundary = visrtx::SmallGeometryBatch::MAX_PRIMITIVES;
  for (const char *type : {"triangle", "sphere", "cone", "cylinder"})
    for (size_t n : {size_t(31),
             size_t(32),
             size_t(33),
             boundary - 1,
             boundary,
             boundary + 1})
      for (bool indexed : {false, true})
        for (bool device : {false, true}) {
          auto g = anari::newObject<anari::Geometry>(d, type);
          auto surface = anari::newObject<anari::Surface>(d);
          anari::setParameter(d, surface, "geometry", g);
          anari::setParameter(d, surface, "material", material);
          anari::commitParameters(d, surface);
          surfaces.push_back(surface);
          cases.push_back({type, n, indexed, device, g, {}});
        }
  anari::release(d, material);
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(
      d, world, "surface", surfaces.data(), surfaces.size());
  anari::commitParameters(d, world);
  for (auto surface : surfaces)
    anari::release(d, surface);
  auto camera = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, camera, "position", vec3{0, 0, -100});
  anari::commitParameters(d, camera);
  auto renderer = anari::newObject<anari::Renderer>(d, "debug");
  anari::commitParameters(d, renderer);
  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", std::array<unsigned, 2>{2, 2});
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(d, frame, "camera", camera);
  anari::setAndReleaseParameter(d, frame, "renderer", renderer);
  anari::setAndReleaseParameter(d, frame, "world", world);
  ANARIFrameCompletionCallback callback = complete;
  anari::setParameter(d,
      frame,
      "frameCompletionCallback",
      ANARI_FRAME_COMPLETION_CALLBACK,
      &callback);
  anari::commitParameters(d, frame);
  bool ok = true;
  for (int generation = 0; generation < 3; ++generation) {
    for (auto &c : cases)
      author(d, c, generation);
    anari::render(d, frame);
    anari::wait(d, frame);
    if (errors || completions != generation + 1) {
      ok = false;
      break;
    }
    for (const auto &c : cases)
      ok = checkPublished(c, generation) && ok;
  }
  anari::release(d, frame);
  for (const auto &c : cases)
    anari::release(d, c.geometry);
  anari::release(d, d);
  ok = errors == 0 && ok;
  std::printf("weighted geometry CDF: %zu cases x 3 generations %s\n",
      cases.size(),
      ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
