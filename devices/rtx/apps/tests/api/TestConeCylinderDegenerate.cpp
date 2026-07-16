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

// GPU regression for degenerate analytic primitives and the cone->cylinder
// continuity limit (intersection-grounds corner cases 4/15/16):
//   - a zero-length cylinder (p0 == p1) and a zero-radius cone (r0 == r1 == 0)
//     must render cleanly: no crash, no NaN/Inf in the framebuffer, and no
//     geometry (they have no surface);
//   - a cone with r0 == r1 must render the same silhouette as a cylinder of
//     that radius (the apex-free quadratic's continuous cylinder limit).

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static constexpr unsigned kRes = 128;

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

// Render one cylinder or cone; returns the FLOAT32 RGBA framebuffer.
static std::vector<vec4> render(anari::Device device,
    const char *subtype,
    const vec3 &p0,
    const vec3 &p1,
    float r0,
    float r1)
{
  auto geometry = anari::newObject<anari::Geometry>(device, subtype);
  {
    if (subtype[0] == 's') { // sphere: single center + radius (p0 is the center)
      anari::setParameterArray1D(
          device, geometry, "vertex.position", &p0, 1);
      anari::setParameter(device, geometry, "radius", r0);
    } else {
      std::array<vec3, 2> positions = {p0, p1};
      anari::setParameterArray1D(device,
          geometry,
          "vertex.position",
          positions.data(),
          positions.size());
      if (subtype[1] == 'o') { // cone: per-vertex radii
        std::array<float, 2> radii = {r0, r1};
        anari::setParameterArray1D(
            device, geometry, "vertex.radius", radii.data(), radii.size());
      } else {
        anari::setParameter(device, geometry, "radius", r0);
      }
      anari::setParameter(device, geometry, "caps", "both");
    }
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

  auto camera = anari::newObject<anari::Camera>(device, "orthographic");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -3.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(device, camera, "aspect", 1.f);
  anari::setParameter(device, camera, "height", 3.f);
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "debug");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "method", "baseColor");
  anari::commitParameters(device, renderer);

  const uvec2 imageSize = {kRes, kRes};
  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  auto fb = anari::map<vec4>(device, frame, "channel.color");
  std::vector<vec4> out(fb.data, fb.data + size_t(fb.width) * fb.height);
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return out;
}

static int coverageCount(const std::vector<vec4> &fb)
{
  int n = 0;
  for (const auto &p : fb)
    n += p[0] > 0.2f ? 1 : 0;
  return n;
}

static bool anyNonFinite(const std::vector<vec4> &fb)
{
  for (const auto &p : fb)
    for (int c = 0; c < 4; ++c)
      if (!std::isfinite(p[c]))
        return true;
  return false;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  int failures = 0;

  const vec3 a{-0.7f, 0.f, 0.f}, b{0.7f, 0.f, 0.f};

  // Degenerate: zero-length cylinder (p0 == p1).
  {
    const auto fb = render(device, "cylinder", a, a, 0.4f, 0.4f);
    if (anyNonFinite(fb)) {
      fprintf(stderr, "FAIL: zero-length cylinder produced NaN/Inf\n");
      ++failures;
    }
    if (coverageCount(fb) != 0) {
      fprintf(stderr, "FAIL: zero-length cylinder rendered geometry\n");
      ++failures;
    }
  }

  // Degenerate: zero-radius cone (both radii zero) — no surface.
  {
    const auto fb = render(device, "cone", a, b, 0.f, 0.f);
    if (anyNonFinite(fb)) {
      fprintf(stderr, "FAIL: zero-radius cone produced NaN/Inf\n");
      ++failures;
    }
    if (coverageCount(fb) != 0) {
      fprintf(stderr, "FAIL: zero-radius cone rendered geometry\n");
      ++failures;
    }
  }

  // Degenerate: zero-radius sphere — no surface (the perp-form discriminant
  // divides the normal by the radius, so a naive solve would emit a NaN normal).
  {
    const auto fb = render(device, "sphere", a, a, 0.f, 0.f);
    if (anyNonFinite(fb)) {
      fprintf(stderr, "FAIL: zero-radius sphere produced NaN/Inf\n");
      ++failures;
    }
    if (coverageCount(fb) != 0) {
      fprintf(stderr, "FAIL: zero-radius sphere rendered geometry\n");
      ++failures;
    }
  }

  // Continuity: a cone with r0 == r1 must match a cylinder of that radius.
  {
    const float r = 0.4f;
    const auto cyl = render(device, "cylinder", a, b, r, r);
    const auto cone = render(device, "cone", a, b, r, r);
    if (anyNonFinite(cone)) {
      fprintf(stderr, "FAIL: equal-radius cone produced NaN/Inf\n");
      ++failures;
    }
    const int cCyl = coverageCount(cyl);
    const int cCone = coverageCount(cone);
    if (cCyl < 200) {
      fprintf(stderr, "FAIL: reference cylinder did not render (cov=%d)\n",
          cCyl);
      ++failures;
    }
    const double rel =
        cCyl > 0 ? double(std::abs(cCyl - cCone)) / cCyl : 1.0;
    if (rel > 0.02) {
      fprintf(stderr,
          "FAIL: equal-radius cone coverage %d vs cylinder %d (%.1f%% off) — "
          "cylinder-limit discontinuity\n",
          cCone, cCyl, 100.0 * rel);
      ++failures;
    }
  }

  anari::release(device, device);

  if (failures) {
    fprintf(stderr, "%d degenerate/continuity failure(s)\n", failures);
    return 1;
  }
  printf("cone/cylinder degenerate + continuity regression passed\n");
  return 0;
}
