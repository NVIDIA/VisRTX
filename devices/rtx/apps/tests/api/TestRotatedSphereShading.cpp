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

// GPU regression for issue #336 (rotated spheres / ellipsoids render black).
// The shading normal of an analytic primitive must be oriented toward the ray
// using a WORLD-space facing test; a prior version reconstructed the
// object-space ray direction from the instance's worldToObject matrix, which
// does not round-trip under a rotation/shear, so the flip boundary was misplaced
// and half the surface got an inward normal and shaded black.
//
// Method: light a sphere with a head-light (the light travels along the view
// direction), so the entire camera-facing hemisphere has a normal that opposes
// the light and MUST be lit. A sphere is rotation-invariant, so rotating (or
// non-uniformly scaling) the instance may not introduce any dark region on the
// visible surface. We assert the covered area is essentially fully lit under
// identity, pure rotation, and rotation + non-uniform scale — the black regions
// of #336 fail this.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>
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
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using mat4 = std::array<float, 16>; // column-major

static constexpr unsigned kRes = 256;

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

// Render a unit sphere under instance transform `xfm`, viewed from `camPos`
// looking at the origin, head-lit (light travels along the view direction so
// the whole camera-facing hemisphere is lit), returning the FLOAT32 RGBA
// framebuffer. The facing bug is view-dependent, so callers orbit `camPos`.
static std::vector<vec4> render(
    anari::Device device, const mat4 &xfm, const vec3 &camPos)
{
  vec3 dir = {-camPos[0], -camPos[1], -camPos[2]};
  {
    const float l =
        std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    dir = {dir[0] / l, dir[1] / l, dir[2] / l};
  }
  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  {
    const vec3 center = {0.f, 0.f, 0.f};
    anari::setParameterArray1D(device, geometry, "vertex.position", &center, 1);
    anari::setParameter(device, geometry, "radius", 1.f);
    anari::commitParameters(device, geometry);
  }

  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", vec3{0.7f, 0.7f, 0.7f});
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  auto group = anari::newObject<anari::Group>(device);
  anari::setParameterArray1D(device, group, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, group);

  auto instance = anari::newObject<anari::Instance>(device, "transform");
  anari::setAndReleaseParameter(device, instance, "group", group);
  anari::setParameter(device, instance, "transform", ANARI_FLOAT32_MAT4,
      xfm.data());
  anari::commitParameters(device, instance);

  // Head-light: travels along the view direction, so the entire camera-facing
  // hemisphere is lit and any dark covered pixel is a wrong-facing normal.
  auto light = anari::newObject<anari::Light>(device, "directional");
  anari::setParameter(device, light, "direction", dir);
  anari::setParameter(device, light, "irradiance", 2.f);
  anari::commitParameters(device, light);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "instance", &instance, 1);
  anari::setParameterArray1D(device, world, "light", &light, 1);
  anari::release(device, instance);
  anari::release(device, light);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "orthographic");
  anari::setParameter(device, camera, "position", camPos);
  anari::setParameter(device, camera, "direction", dir);
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(device, camera, "aspect", 1.f);
  anari::setParameter(device, camera, "height", 2.6f);
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "default");
  // Distinct blue background so a covered-but-BLACK pixel (a flipped normal,
  // ~(0,0,0)) is still separable from the background by color distance.
  anari::setParameter(
      device, renderer, "background", vec4{0.1f, 0.35f, 0.6f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", 16);
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

static mat4 identity()
{
  return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}

// column-major, rows = images of the basis vectors
static mat4 linear(const vec3 &cx, const vec3 &cy, const vec3 &cz)
{
  return {cx[0], cx[1], cx[2], 0, cy[0], cy[1], cy[2], 0, cz[0], cz[1], cz[2], 0,
      0, 0, 0, 1};
}

// Fraction of covered pixels that are DARK (unlit). Coverage is detected by
// color distance from the (distinct blue) background, so a covered-but-black
// pixel — the signature of a flipped/inward normal — counts as covered rather
// than being mistaken for background.
static double darkCoveredFraction(const std::vector<vec4> &fb, int &coveredOut)
{
  const vec3 bg = {0.1f, 0.35f, 0.6f};
  int covered = 0, dark = 0;
  for (const auto &p : fb) {
    const float dr = p[0] - bg[0], dg = p[1] - bg[1], db = p[2] - bg[2];
    const bool isCov = (dr * dr + dg * dg + db * db) > 0.02f;
    if (!isCov)
      continue;
    ++covered;
    const float luma = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
    if (luma < 0.02f)
      ++dark;
  }
  coveredOut = covered;
  return covered > 0 ? double(dark) / covered : 1.0;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const float c = std::cos(0.7f), s = std::sin(0.7f);
  struct Case
  {
    const char *name;
    mat4 m;
  };
  const std::array<Case, 6> cases = {
      Case{"identity", identity()},
      Case{"rotation-Z", linear({c, s, 0}, {-s, c, 0}, {0, 0, 1})},
      // rotation about a general axis (the #336 trigger)
      Case{"rotation-diag",
          linear({0.804f, 0.506f, -0.313f}, {-0.310f, 0.804f, 0.506f},
              {0.506f, -0.313f, 0.804f})},
      // rotation + non-uniform scale (ellipsoid)
      Case{"rot+nonuniform",
          linear({1.6f * c, 1.6f * s, 0}, {-0.5f * s, 0.5f * c, 0},
              {0, 0, 1})},
      // shear (non-orthonormal, non-symmetric): the world-space facing test
      // must handle it too — this is the signed-facing-under-shear gate for the
      // shared analytic-primitive normal path (sphere/cylinder/cone).
      Case{"shear", linear({1, 0, 0}, {0.7f, 1, 0}, {0.3f, -0.5f, 1})},
      // rotation composed with shear
      Case{"rot+shear",
          linear({c, s, 0}, {0.7f * c - s, 0.7f * s + c, 0}, {0, 0, 1})}};

  // The facing bug is view-dependent, so orbit the camera; the head-light
  // follows, so from every angle the visible hemisphere must be fully lit.
  const double R = 3.0;
  std::array<vec3, 6> orbit;
  for (int i = 0; i < 6; ++i) {
    const double a = 2.0 * M_PI * i / 6.0;
    orbit[i] = {float(R * std::sin(a)), 0.6f, float(-R * std::cos(a))};
  }

  int failures = 0;
  for (const auto &cs : cases) {
    for (size_t v = 0; v < orbit.size(); ++v) {
      int covered = 0;
      const auto fb = render(device, cs.m, orbit[v]);
      const double darkFrac = darkCoveredFraction(fb, covered);
      if (covered < 2000) {
        fprintf(stderr,
            "FAIL: %s view %zu under-covered (%d px) — did not render\n",
            cs.name, v, covered);
        ++failures;
        continue;
      }
      // A head-lit sphere's visible hemisphere is fully lit; only a thin
      // silhouette rim (grazing, N.L -> 0) may dim. Allow a few percent for
      // that and AA; a wrong-facing normal blackens a large fraction.
      if (darkFrac > 0.05) {
        fprintf(stderr,
            "FAIL: %s view %zu has %.1f%% dark covered pixels — inward/flipped "
            "normal (issue #336)\n",
            cs.name, v, 100.0 * darkFrac);
        ++failures;
      }
    }
  }

  anari::release(device, device);

  if (failures) {
    fprintf(stderr, "%d rotated-sphere shading failure(s)\n", failures);
    return 1;
  }
  printf("rotated sphere / ellipsoid shading regression passed (#336)\n");
  return 0;
}
