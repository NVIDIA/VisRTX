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

// Quad light emission side. The ANARI spec defines the quad light's front side
// as edge2 x edge1. A quad light above a floor with edge1 = +X, edge2 = +Z has
// its front facing +Y (away from the floor), so 'front' must leave the floor
// dark and 'back' must light it; swapping the edges flips both outcomes.

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

static constexpr uvec2 IMAGE_SIZE = {64, 64};
static constexpr int PIXEL_SAMPLES = 16;

static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;

static anari::Surface makeFloor(ANARIDevice device)
{
  const std::array<vec3, 4> pos = {vec3{-6.f, 0.f, -6.f},
      vec3{6.f, 0.f, -6.f},
      vec3{6.f, 0.f, 6.f},
      vec3{-6.f, 0.f, 6.f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, mat, "color", vec3{0.8f, 0.8f, 0.8f});
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// Quad light above the floor; 'swapEdges' exchanges edge1 and edge2.
static anari::Light makeQuadLight(
    ANARIDevice device, const char *side, bool swapEdges)
{
  const vec3 ex = {2.f * QUAD_HALF, 0.f, 0.f};
  const vec3 ez = {0.f, 0.f, 2.f * QUAD_HALF};

  auto light = anari::newObject<anari::Light>(device, "quad");
  anari::setParameter(
      device, light, "position", vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF});
  anari::setParameter(device, light, "edge1", swapEdges ? ez : ex);
  anari::setParameter(device, light, "edge2", swapEdges ? ex : ez);
  anari::setParameter(device, light, "intensity", 8.f);
  anari::setParameter(device, light, "side", side);
  anari::commitParameters(device, light);
  return light;
}

// Mean luminance of the floor seen from above, lit only by the quad light.
static double renderFloor(ANARIDevice device, const char *side, bool swapEdges)
{
  auto surface = makeFloor(device);
  auto light = makeQuadLight(device, side, swapEdges);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::setParameterArray1D(device, world, "light", &light, 1);
  anari::release(device, surface);
  anari::release(device, light);
  anari::commitParameters(device, world);

  // Look down at the floor from beside the light so the light itself is never
  // in view; only its contribution to the floor is measured.
  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 1.f, -2.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, -1.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", PIXEL_SAMPLES);
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
  const uint64_t n = uint64_t(fb.width) * fb.height;
  for (uint64_t i = 0; i < n; ++i) {
    const vec4 &p = fb.data[i];
    sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
  }
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return n ? sum / double(n) : 0.0;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  struct Case
  {
    const char *side;
    bool swapEdges;
    bool expectLit;
  };
  // edge2 x edge1 = +Z x +X = +Y (up, away from the floor) when not swapped.
  const Case cases[] = {
      {"front", false, false},
      {"back", false, true},
      {"front", true, true},
      {"back", true, false},
      {"both", false, true},
  };

  bool ok = true;
  for (const auto &c : cases) {
    const double lum = renderFloor(device, c.side, c.swapEdges);
    const bool lit = lum > 1e-3;
    printf("side=%-5s swapEdges=%d  floorLuminance=%f  (%s)\n",
        c.side,
        int(c.swapEdges),
        lum,
        lit == c.expectLit ? "ok" : "WRONG");
    ok = ok && lit == c.expectLit;
  }

  anari::release(device, device);

  if (!ok) {
    fprintf(stderr, "FAIL: quad light emits from the wrong side\n");
    return 1;
  }
  printf("quad light side test passed\n");
  return 0;
}
