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

// ANARI 1.2 alpha-mode resolution on matte and physicallyBased materials. With
// no "alphaMode" string set, the mode derives from "alphaCutoff": unset means
// blend, 1 means opaque, and < 1 means mask. An explicit "alphaMode" string
// (ANARI <= 1.1) still takes precedence.
//
// Method: a black quad with opacity 0.5 fills the center of the view over a
// WHITE background. The center pixel then reads ~0 when the quad is opaque (or
// masked in), ~0.5 when blended, and ~1 when masked out. Each case is set on a
// single material and re-committed, which also checks that removing a
// parameter restores the derived mode (no stale state).

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <cstdio>
#include <cstdlib>
#include <optional>

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static constexpr unsigned kRes = 64;

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

enum class Expect
{
  COVERED, // opaque, or masked in
  BLENDED,
  CLEAR // masked out
};

struct Case
{
  const char *name;
  const char *alphaMode; // nullptr = unset
  std::optional<float> alphaCutoff; // nullopt = unset
  Expect expect;
};

static float luma(const vec4 &p)
{
  return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
}

static int runMaterial(anari::Device device, const char *subtype)
{
  auto material = anari::newObject<anari::Material>(device, subtype);
  const bool isMatte = subtype[0] == 'm';
  anari::setParameter(
      device, material, isMatte ? "color" : "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(device, material, "opacity", 0.5f);
  if (!isMatte) {
    anari::setParameter(device, material, "metallic", 0.f);
    anari::setParameter(device, material, "roughness", 1.f);
  }
  anari::commitParameters(device, material);

  const std::array<vec3, 4> positions = {vec3{-1.f, -1.f, 0.f},
      vec3{1.f, -1.f, 0.f},
      vec3{1.f, 1.f, 0.f},
      vec3{-1.f, 1.f, 0.f}};
  auto geometry = anari::newObject<anari::Geometry>(device, "quad");
  anari::setParameterArray1D(
      device, geometry, "vertex.position", positions.data(), positions.size());
  anari::commitParameters(device, geometry);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "orthographic");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -3.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(device, camera, "aspect", 1.f);
  anari::setParameter(device, camera, "height", 1.f);
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "background", vec4{1.f, 1.f, 1.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 1.f);
  anari::setParameter(device, renderer, "pixelSamples", 16);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", uvec2{kRes, kRes});
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  const std::array<Case, 9> cases = {
      Case{"default (cutoff unset)", nullptr, std::nullopt, Expect::BLENDED},
      Case{"cutoff=1", nullptr, 1.f, Expect::COVERED},
      Case{"cutoff=0.3", nullptr, 0.3f, Expect::COVERED},
      Case{"cutoff=0.7", nullptr, 0.7f, Expect::CLEAR},
      Case{"cutoff=opacity", nullptr, 0.5f, Expect::COVERED},
      Case{"cutoff removed", nullptr, std::nullopt, Expect::BLENDED},
      Case{"legacy opaque", "opaque", std::nullopt, Expect::COVERED},
      Case{"legacy blend + cutoff=1", "blend", 1.f, Expect::BLENDED},
      Case{"legacy mask + cutoff=1", "mask", 1.f, Expect::CLEAR},
  };

  int failures = 0;
  for (const auto &c : cases) {
    if (c.alphaMode)
      anari::setParameter(device, material, "alphaMode", c.alphaMode);
    else
      anari::unsetParameter(device, material, "alphaMode");
    if (c.alphaCutoff)
      anari::setParameter(device, material, "alphaCutoff", *c.alphaCutoff);
    else
      anari::unsetParameter(device, material, "alphaCutoff");
    anari::commitParameters(device, material);

    anari::render(device, frame);
    anari::wait(device, frame);
    auto fb = anari::map<vec4>(device, frame, "channel.color");
    const float L = luma(fb.data[(kRes / 2) * kRes + kRes / 2]);
    anari::unmap(device, frame, "channel.color");

    bool ok = false;
    switch (c.expect) {
    case Expect::COVERED:
      ok = L < 0.15f;
      break;
    case Expect::BLENDED:
      ok = L > 0.3f && L < 0.7f;
      break;
    case Expect::CLEAR:
      ok = L > 0.9f;
      break;
    }
    if (!ok || getenv("ALPHA_DEBUG"))
      fprintf(stderr,
          "%s %s: %s center luma=%.3f\n",
          ok ? "ok" : "FAIL:",
          subtype,
          c.name,
          L);
    failures += !ok;
  }

  anari::release(device, frame);
  anari::release(device, material);
  return failures;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  int failures = runMaterial(device, "matte");
  failures += runMaterial(device, "physicallyBased");
  anari::release(device, device);

  if (failures) {
    fprintf(stderr, "%d alpha mode resolution failure(s)\n", failures);
    return 1;
  }
  printf("alpha mode resolution passed\n");
  return 0;
}
