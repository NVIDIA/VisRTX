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

// A Geometry Light must light the scene in the 'interactive' renderer too — it
// joins the shared light-instance list, so interactive's all-lights NEE loop
// should sample it. Renders a floor lit only by an emissive quad with both
// 'quality' and 'interactive' and asserts the floor pool is lit in each.

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
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

static constexpr uvec2 IMAGE_SIZE = {256, 256};
static constexpr float EMISSIVE_RADIANCE = 8.f;
static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;

static anari::Surface triangleSurface(ANARIDevice device,
    const std::array<vec3, 4> &pos,
    anari::Material mat)
{
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};
  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);
  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

static double render(ANARIDevice device, const char *rendererSubtype)
{
  auto floorMat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, floorMat, "baseColor", vec3{0.6f, 0.6f, 0.6f});
  anari::commitParameters(device, floorMat);
  auto floor = triangleSurface(device,
      {vec3{-6.f, 0.f, -6.f},
          vec3{6.f, 0.f, -6.f},
          vec3{6.f, 0.f, 6.f},
          vec3{-6.f, 0.f, 6.f}},
      floorMat);

  auto emMat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, emMat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(device,
      emMat,
      "emissive",
      vec3{EMISSIVE_RADIANCE, EMISSIVE_RADIANCE, EMISSIVE_RADIANCE});
  anari::commitParameters(device, emMat);
  auto quad = triangleSurface(device,
      {vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF},
          vec3{QUAD_HALF, QUAD_Y, -QUAD_HALF},
          vec3{QUAD_HALF, QUAD_Y, QUAD_HALF},
          vec3{-QUAD_HALF, QUAD_Y, QUAD_HALF}},
      emMat);

  const std::array<anari::Surface, 2> surfaces = {floor, quad};
  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", surfaces.data(), 2);
  anari::release(device, floor);
  anari::release(device, quad);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.5f, -3.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, -0.15f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, rendererSubtype);
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", 64);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", IMAGE_SIZE);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);
  auto fb = anari::map<vec4>(device, frame, "channel.color");

  double sum = 0.0;
  uint64_t n = 0;
  for (uint32_t y = IMAGE_SIZE[1] / 8; y < IMAGE_SIZE[1] / 2; ++y) {
    for (uint32_t x = 3 * IMAGE_SIZE[0] / 8; x < 5 * IMAGE_SIZE[0] / 8; ++x) {
      const vec4 &p = fb.data[y * IMAGE_SIZE[0] + x];
      sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
      ++n;
    }
  }
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return n ? sum / double(n) : 0.0;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  const double quality = render(device, "quality");
  const double interactive = render(device, "interactive");
  anari::release(device, device);

  printf("quality floor=%f  interactive floor=%f\n", quality, interactive);

  if (quality <= 0.01) {
    fprintf(stderr, "FAIL: geometry light did not light the floor in quality\n");
    return 1;
  }
  // The interactive floor pool must be lit by the Geometry Light — a dark pool
  // means it is not participating in interactive's direct lighting.
  if (interactive <= 0.01) {
    fprintf(stderr,
        "FAIL: geometry light does not light the floor in interactive "
        "(interactive=%f)\n",
        interactive);
    return 1;
  }
  printf("emissive geometry light interactive lighting passed\n");
  return 0;
}
