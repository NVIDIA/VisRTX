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

// GPU regression for back-face (exit) crossings of the analytic primitives:
// a camera placed INSIDE a sphere, capped cylinder, and cone must see the
// primitive's interior (previously the IS programs reported only front-facing
// entry crossings, so interiors rendered as background). An outside view of
// the sphere is rendered as a control.

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

static anari::Geometry makeSphere(anari::Device device)
{
  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  const vec3 center = {0.f, 0.f, 0.f};
  anari::setParameterArray1D(device, geometry, "vertex.position", &center, 1);
  anari::setParameter(device, geometry, "radius", 1.f);
  anari::commitParameters(device, geometry);
  return geometry;
}

static anari::Geometry makeCylinder(anari::Device device)
{
  auto geometry = anari::newObject<anari::Geometry>(device, "cylinder");
  std::array<vec3, 2> positions = {vec3{0.f, 0.f, -1.f}, vec3{0.f, 0.f, 1.f}};
  anari::setParameterArray1D(
      device, geometry, "vertex.position", positions.data(), positions.size());
  anari::setParameter(device, geometry, "radius", 1.f);
  anari::setParameter(device, geometry, "caps", "both");
  anari::commitParameters(device, geometry);
  return geometry;
}

static anari::Geometry makeCone(anari::Device device)
{
  auto geometry = anari::newObject<anari::Geometry>(device, "cone");
  std::array<vec3, 2> positions = {vec3{0.f, 0.f, -1.f}, vec3{0.f, 0.f, 1.f}};
  std::array<float, 2> radii = {1.5f, 0.5f};
  anari::setParameterArray1D(
      device, geometry, "vertex.position", positions.data(), positions.size());
  anari::setParameterArray1D(
      device, geometry, "vertex.radius", radii.data(), radii.size());
  anari::setParameter(device, geometry, "caps", "both");
  anari::commitParameters(device, geometry);
  return geometry;
}

// Render the geometry with the camera at `eye` looking down +z and return
// whether the center pixel is covered by geometry (non-background).
static bool centerPixelHit(
    anari::Device device, anari::Geometry geometry, const vec3 &eye)
{
  const uvec2 imageSize = {64, 64};
  const vec4 background = {0.f, 0.f, 0.f, 1.f};

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

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", eye);
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
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  auto fb = anari::map<uint32_t>(device, frame, "channel.color");
  const uint32_t center =
      fb.data[(imageSize[1] / 2) * fb.width + imageSize[0] / 2];
  const uint32_t red = center & 0xFFu; // R channel (RGBA8, little-endian)
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);

  return red > 16u; // geometry (red) vs black background
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  struct Case
  {
    const char *name;
    bool hit;
    bool expected;
  };
  const vec3 inside = {0.f, 0.f, 0.f};
  const vec3 outside = {0.f, 0.f, -3.f};
  const Case cases[] = {
      {"sphere exterior (control)",
          centerPixelHit(device, makeSphere(device), outside),
          true},
      {"sphere interior",
          centerPixelHit(device, makeSphere(device), inside),
          true},
      {"cylinder interior",
          centerPixelHit(device, makeCylinder(device), inside),
          true},
      {"cone interior", centerPixelHit(device, makeCone(device), inside), true},
  };

  anari::release(device, device);

  int failures = 0;
  for (const auto &c : cases) {
    if (c.hit != c.expected) {
      fprintf(stderr,
          "FAIL: %s — expected %s, got %s\n",
          c.name,
          c.expected ? "hit" : "miss",
          c.hit ? "hit" : "miss");
      ++failures;
    }
  }
  if (failures)
    return 1;
  printf("analytic primitive interior (back-face) regression passed\n");
  return 0;
}
