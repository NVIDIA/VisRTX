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

// GPU regression for analytic sphere/cylinder/cone intersection under the
// transform threat model (issues #73/#333/#336 corner cases). Two properties,
// per prim:
//
//   A. Linear-transform resilience (rotation, non-uniform scale, shear): the
//      rendered silhouette is watertight (no interior speckle/fray) and every
//      covered pixel carries a finite, non-degenerate world-space normal. This
//      exercises the object-space intersection + inverse-transpose normal path
//      through a non-orthonormal OptiX instance transform end to end.
//
//   B. Far/tiny/huge OBJECT-space coordinates: the intersector's origin
//      re-centering + conditioned discriminant must make a primitive whose
//      vertices sit at ~1e4 (or ~1e-3) render identically to the same primitive
//      at unit scale near the origin. The geometry vertices themselves are
//      placed far/scaled (NOT via an instance transform, which would absorb the
//      offset back into object space), with the camera moved to match, so the
//      intersector genuinely sees the extreme coordinates.

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

static constexpr unsigned kRes = 200;

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

struct Prim
{
  const char *subtype; // "sphere", "cylinder" or "cone"
  vec3 p0, p1; // sphere uses p0 as the center
  float r0, r1; // sphere/cylinder use r0 as the single radius
};

// Render one primitive (optionally under an instance transform) with the debug
// renderer, returning the FLOAT32 RGBA framebuffer. `method` selects baseColor
// (flat silhouette) or Ng.abs (|world normal| per channel).
static std::vector<vec4> render(anari::Device device,
    const Prim &prim,
    const mat4 *xfm,
    const vec3 &camPos,
    const vec3 &camDir,
    float camScale, // orthographic height
    const char *method)
{
  auto geometry = anari::newObject<anari::Geometry>(device, prim.subtype);
  {
    if (prim.subtype[0] == 's') { // sphere: single center + radius
      anari::setParameterArray1D(
          device, geometry, "vertex.position", &prim.p0, 1);
      anari::setParameter(device, geometry, "radius", prim.r0);
    } else {
      std::array<vec3, 2> positions = {prim.p0, prim.p1};
      anari::setParameterArray1D(device,
          geometry,
          "vertex.position",
          positions.data(),
          positions.size());
      if (prim.subtype[1] == 'o') { // cone
        std::array<float, 2> radii = {prim.r0, prim.r1};
        anari::setParameterArray1D(
            device, geometry, "vertex.radius", radii.data(), radii.size());
      } else {
        anari::setParameter(device, geometry, "radius", prim.r0);
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

  auto group = anari::newObject<anari::Group>(device);
  anari::setParameterArray1D(device, group, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, group);

  auto instance = anari::newObject<anari::Instance>(device, "transform");
  anari::setAndReleaseParameter(device, instance, "group", group);
  if (xfm)
    anari::setParameter(
        device, instance, "transform", ANARI_FLOAT32_MAT4, xfm->data());
  anari::commitParameters(device, instance);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "instance", &instance, 1);
  anari::release(device, instance);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "orthographic");
  anari::setParameter(device, camera, "position", camPos);
  anari::setParameter(device, camera, "direction", camDir);
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(device, camera, "aspect", 1.f);
  anari::setParameter(device, camera, "height", camScale);
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "debug");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "method", method);
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

static bool covered(const vec4 &p) // geometry (red) vs black background
{
  return p[0] > 0.2f;
}

static bool validNormal(const vec4 &p) // Ng.abs: |N| has a dominant component
{
  const float m = std::max(p[0], std::max(p[1], p[2]));
  return m > 0.2f && std::isfinite(p[0]) && std::isfinite(p[1])
      && std::isfinite(p[2]);
}

static int coverageCount(const std::vector<vec4> &fb)
{
  int n = 0;
  for (const auto &p : fb)
    n += covered(p) ? 1 : 0;
  return n;
}

// Interior speckle: a background pixel with geometry on all four sides (within
// the row/column). For a convex silhouette — which the affine image of a
// cylinder/cone always is — any such pixel is a genuine hole (fray).
static int interiorHoles(const std::vector<vec4> &fb)
{
  int holes = 0;
  for (unsigned y = 0; y < kRes; ++y) {
    for (unsigned x = 0; x < kRes; ++x) {
      if (covered(fb[y * kRes + x]))
        continue;
      bool l = false, r = false, u = false, d = false;
      for (unsigned i = 0; i < x; ++i)
        l |= covered(fb[y * kRes + i]);
      for (unsigned i = x + 1; i < kRes; ++i)
        r |= covered(fb[y * kRes + i]);
      for (unsigned i = 0; i < y; ++i)
        u |= covered(fb[i * kRes + x]);
      for (unsigned i = y + 1; i < kRes; ++i)
        d |= covered(fb[i * kRes + x]);
      if (l && r && u && d)
        ++holes;
    }
  }
  return holes;
}

static mat4 identity()
{
  return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}

// Column-major 3x3 linear part (last column translation), rows given as the
// image of the basis vectors.
static mat4 linear(const vec3 &cx, const vec3 &cy, const vec3 &cz)
{
  return {cx[0], cx[1], cx[2], 0, cy[0], cy[1], cy[2], 0, cz[0], cz[1], cz[2], 0,
      0, 0, 0, 1};
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const vec3 camDir = {0.f, 0.f, 1.f};
  int failures = 0;

  // A unit primitive centered on the origin, framed by a z-looking ortho camera.
  const std::array<Prim, 3> units = {
      Prim{"sphere", {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, 0.5f, 0.f},
      Prim{"cylinder", {-0.7f, 0.f, 0.f}, {0.7f, 0.f, 0.f}, 0.35f, 0.35f},
      Prim{"cone", {0.f, -0.7f, 0.f}, {0.f, 0.7f, 0.f}, 0.4f, 0.1f}};

  // ---- Part A: linear-transform resilience -------------------------------
  struct Xf
  {
    const char *name;
    mat4 m;
  };
  const float c = std::cos(0.7f), s = std::sin(0.7f);
  const std::array<Xf, 4> xforms = {
      Xf{"rotation", linear({c, s, 0}, {-s, c, 0}, {0, 0, 1})},
      Xf{"nonuniform-scale", linear({1.8f, 0, 0}, {0, 0.5f, 0}, {0, 0, 1})},
      // rotation composed with non-uniform scale (a non-symmetric linear map)
      Xf{"rot+scale",
          linear({1.8f * c, 1.8f * s, 0}, {-0.5f * s, 0.5f * c, 0}, {0, 0, 1})},
      // shear: x <- x + 0.6 y (non-orthonormal, off-diagonal)
      Xf{"shear", linear({1, 0, 0}, {0.6f, 1, 0}, {0, 0, 1})}};

  for (const auto &prim : units) {
    for (const auto &xf : xforms) {
      const auto base = render(
          device, prim, &xf.m, {0, 0, -3}, camDir, 3.f, "baseColor");
      const auto norm =
          render(device, prim, &xf.m, {0, 0, -3}, camDir, 3.f, "Ng.abs");
      const int cov = coverageCount(base);
      const int holes = interiorHoles(base);
      int validCov = 0;
      for (size_t i = 0; i < base.size(); ++i)
        if (covered(base[i]) && validNormal(norm[i]))
          ++validCov;

      if (cov < 200) {
        fprintf(stderr, "FAIL: %s %s vanished (coverage=%d)\n", prim.subtype,
            xf.name, cov);
        ++failures;
      }
      if (holes > 0) {
        fprintf(stderr,
            "FAIL: %s %s silhouette has %d interior hole(s) (fray under "
            "transform)\n",
            prim.subtype, xf.name, holes);
        ++failures;
      }
      if (cov > 0 && validCov < 0.97 * cov) {
        fprintf(stderr,
            "FAIL: %s %s only %d/%d covered pixels have a valid normal "
            "(degenerate normals under transform)\n",
            prim.subtype, xf.name, validCov, cov);
        ++failures;
      }
    }
  }

  // ---- Part B: far / tiny / huge object-space coordinates ----------------
  // Reference: unit prim at the origin, no transform.
  for (const auto &unit : units) {
    const auto ref =
        render(device, unit, nullptr, {0, 0, -3}, camDir, 3.f, "baseColor");
    const int refCov = coverageCount(ref);

    struct Sc
    {
      const char *name;
      float k;
    };
    // Place the vertices (and camera) at k times unit scale so the intersector
    // sees object-space coordinates of magnitude ~k; the image must be
    // invariant (origin re-centering + conditioned discriminant).
    const std::array<Sc, 3> scales = {
        Sc{"far-8e3", 8000.f}, Sc{"tiny-1e-3", 1e-3f}, Sc{"huge-1e4", 1e4f}};
    for (const auto &sc : scales) {
      const float k = sc.k;
      // Offset far from the origin AND scale, so both magnitude regimes hit.
      const vec3 off = {k, -k, k};
      Prim p = unit;
      p.p0 = {off[0] + unit.p0[0] * k, off[1] + unit.p0[1] * k,
          off[2] + unit.p0[2] * k};
      p.p1 = {off[0] + unit.p1[0] * k, off[1] + unit.p1[1] * k,
          off[2] + unit.p1[2] * k};
      p.r0 = unit.r0 * k;
      p.r1 = unit.r1 * k;
      const vec3 camPos = {off[0], off[1], off[2] - 3.f * k};
      const auto img =
          render(device, p, nullptr, camPos, camDir, 3.f * k, "baseColor");
      const int cov = coverageCount(img);
      const int holes = interiorHoles(img);
      // Coverage must match the unit reference within a small tolerance
      // (identical geometry up to a global similarity transform).
      const double rel =
          refCov > 0 ? double(std::abs(cov - refCov)) / refCov : 1.0;
      if (rel > 0.06) {
        fprintf(stderr,
            "FAIL: %s %s coverage %d vs reference %d (%.1f%% off) — "
            "conditioning breaks at extreme coordinates\n",
            unit.subtype, sc.name, cov, refCov, 100.0 * rel);
        ++failures;
      }
      if (holes > 0) {
        fprintf(stderr, "FAIL: %s %s has %d interior hole(s) at scale %g\n",
            unit.subtype, sc.name, holes, k);
        ++failures;
      }
    }
  }

  anari::release(device, device);

  if (failures) {
    fprintf(stderr, "%d transform-resilience failure(s)\n", failures);
    return 1;
  }
  printf("cone/cylinder transform + far-coordinate regression passed\n");
  return 0;
}
