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

// GPU regression for the capped-cylinder end-on bug: a cylinder viewed straight
// down its axis must show its flat cap (previously the axis-parallel ray hit
// nothing). Renders end-on with caps="both" (center pixel must be geometry) and
// caps="none" (center pixel must be background), asserting both.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>
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

// Render a single unit cylinder end-on and return whether the center pixel is
// covered by geometry (non-background).
static bool centerPixelHit(anari::Device device, const char *caps)
{
  const uvec2 imageSize = {64, 64};
  const vec4 background = {0.f, 0.f, 0.f, 1.f};

  auto geometry = anari::newObject<anari::Geometry>(device, "cylinder");
  {
    // One cylinder along +z, [0,1], fat enough to cover the center pixel.
    std::array<vec3, 2> positions = {vec3{0.f, 0.f, 0.f}, vec3{0.f, 0.f, 1.f}};
    anari::setParameterArray1D(
        device, geometry, "vertex.position", positions.data(), positions.size());
    anari::setParameter(device, geometry, "radius", 0.4f);
    anari::setParameter(device, geometry, "caps", caps);
    anari::commitParameters(device, geometry);
  }

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

  // Camera looks straight down the +z axis at the near cap.
  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -3.f});
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
  const uint32_t center = fb.data[(imageSize[1] / 2) * fb.width + imageSize[0] / 2];
  const uint32_t red = center & 0xFFu; // R channel (RGBA8, little-endian)
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);

  return red > 16u; // geometry (red) vs black background
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const bool cappedHit = centerPixelHit(device, "both");
  const bool uncappedHit = centerPixelHit(device, "none");

  anari::release(device, device);

  int failures = 0;
  if (!cappedHit) {
    fprintf(stderr, "FAIL: end-on caps=both cylinder shows no cap (regression)\n");
    ++failures;
  }
  if (uncappedHit) {
    fprintf(stderr, "FAIL: end-on caps=none cylinder unexpectedly hit\n");
    ++failures;
  }
  if (failures)
    return 1;
  printf("cylinder caps end-on regression passed\n");
  return 0;
}
