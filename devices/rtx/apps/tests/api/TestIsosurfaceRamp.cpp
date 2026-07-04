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

// GPU end-to-end test for the isosurface intersector (DDA + macrocell-skip +
// analytic linear crossing). A structuredRegular field holds a z-ramp
// (value == z), so an isovalue selects a z-plane; a camera at z=-5 looking down
// +z through the plane center lets us pin the center-pixel hit/miss and the
// center depth (distance from camera to the plane). Covers: linear filter hit,
// nearest filter hit, multi-isovalue nearest-wins, and out-of-range miss.

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
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using ivec3 = std::array<int, 3>;
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

struct CenterProbe
{
  bool hit;
  float depth;
};

// Render the z-ramp isosurface and report the center-pixel hit + depth. A
// single isovalue is set as a scalar FLOAT32 param; multiple isovalues use a
// FLOAT32 Array1D — both parameter forms are exercised.
static CenterProbe centerProbe(anari::Device device,
    const char *filter,
    const std::vector<float> &isovalues)
{
  const uvec2 imageSize = {64, 64};
  const vec4 background = {0.f, 0.f, 0.f, 1.f};
  const ivec3 dims = {8, 8, 8};

  // Field data: a linear z-ramp, data[i,j,k] = float(k). ANARI Array3D is
  // stored x-fastest.
  std::vector<float> fieldData(size_t(dims[0]) * dims[1] * dims[2]);
  for (int k = 0; k < dims[2]; ++k)
    for (int j = 0; j < dims[1]; ++j)
      for (int i = 0; i < dims[0]; ++i)
        fieldData[size_t(i) + dims[0] * (size_t(j) + size_t(dims[1]) * k)] =
            float(k);

  auto field =
      anari::newObject<anari::SpatialField>(device, "structuredRegular");
  {
    auto data =
        anari::newArray3D(device, fieldData.data(), dims[0], dims[1], dims[2]);
    anari::setAndReleaseParameter(device, field, "data", data);
    anari::setParameter(device, field, "origin", vec3{0.f, 0.f, 0.f});
    anari::setParameter(device, field, "spacing", vec3{1.f, 1.f, 1.f});
    anari::setParameter(device, field, "filter", filter);
    // Commit the field BEFORE the geometry that references it: the isosurface
    // finalize needs the field's space-skipping grid, otherwise it warns
    // "space-skipping grid not ready" and renders nothing.
    anari::commitParameters(device, field);
  }

  auto geometry = anari::newObject<anari::Geometry>(device, "isosurface");
  anari::setParameter(device, geometry, "field", field);
  if (isovalues.size() == 1)
    anari::setParameter(device, geometry, "isovalue", isovalues[0]);
  else
    anari::setParameterArray1D(
        device, geometry, "isovalue", isovalues.data(), isovalues.size());
  anari::commitParameters(device, geometry);
  anari::release(device, field);

  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", vec3{1.f, 0.f, 0.f});
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  // Camera at z=-5 looking down +z at the center of the iso-plane.
  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{3.5f, 3.5f, -5.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "debug");
  anari::setParameter(device, renderer, "background", background);
  anari::setParameter(device, renderer, "method", "baseColor");
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(device, frame, "channel.depth", ANARI_FLOAT32);
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  const size_t centerIdx =
      (imageSize[1] / 2) * size_t(imageSize[0]) + imageSize[0] / 2;

  auto fb = anari::map<uint32_t>(device, frame, "channel.color");
  const uint32_t center = fb.data[centerIdx];
  const uint32_t red = center & 0xFFu; // R channel (RGBA8, little-endian)
  anari::unmap(device, frame, "channel.color");

  auto depthBuf = anari::map<float>(device, frame, "channel.depth");
  const float depth = depthBuf.data ? depthBuf.data[centerIdx] : -1.f;
  anari::unmap(device, frame, "channel.depth");

  anari::release(device, frame);

  return {red > 16u, depth}; // geometry (red) vs black background
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const CenterProbe linearHit = centerProbe(device, "linear", {3.5f});
  const CenterProbe nearestHit = centerProbe(device, "nearest", {3.5f});
  const CenterProbe multiHit = centerProbe(device, "linear", {1.5f, 3.5f});
  const CenterProbe miss = centerProbe(device, "linear", {100.f});

  anari::release(device, device);

  const float kDepthTol = 0.15f;
  int failures = 0;

  // 1. Linear filter, isovalue 3.5: plane at z=3.5, camera z=-5 -> depth 8.5.
  if (!linearHit.hit) {
    fprintf(stderr, "FAIL: linear-filter isovalue 3.5 center is background\n");
    ++failures;
  } else if (std::fabs(linearHit.depth - 8.5f) > kDepthTol) {
    fprintf(stderr,
        "FAIL: linear-filter depth %.3f, expected ~8.5\n",
        linearHit.depth);
    ++failures;
  }

  // 2. Nearest filter, isovalue 3.5: center must be geometry (depth loose).
  if (!nearestHit.hit) {
    fprintf(stderr, "FAIL: nearest-filter isovalue 3.5 center is background\n");
    ++failures;
  }

  // 3. Multi-isovalue {1.5, 3.5}, linear: nearest plane (z=1.5) wins -> 6.5.
  if (!multiHit.hit) {
    fprintf(stderr, "FAIL: multi-isovalue center is background\n");
    ++failures;
  } else if (std::fabs(multiHit.depth - 6.5f) > kDepthTol) {
    fprintf(stderr,
        "FAIL: multi-isovalue depth %.3f, expected ~6.5 (nearest plane)\n",
        multiHit.depth);
    ++failures;
  }

  // 4. Isovalue 100 is outside the field range -> center is background.
  if (miss.hit) {
    fprintf(stderr, "FAIL: out-of-range isovalue 100 unexpectedly hit\n");
    ++failures;
  }

  if (failures)
    return 1;
  printf("isosurface z-ramp intersector test passed\n");
  return 0;
}
