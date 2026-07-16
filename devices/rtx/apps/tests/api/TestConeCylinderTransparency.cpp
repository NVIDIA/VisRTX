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

// GPU regression for the grazing PHANTOM-EXIT dark ring on transparent analytic
// primitives (the kGrazeRelEps guard; intersection-grounds corner case 20).
// Near the silhouette a fp32-noise discriminant can spawn a spurious back-facing
// EXIT crossing at t up to ~1e-3*r beyond tangency; on a transparent (blend) or
// shadow-opacity anyhit that extra crossing over-attenuates a multi-pixel band,
// producing a dark ring around the silhouette. The grazing-discriminant gate
// (kGrazeRelEps) suppresses it.
//
// Method: render a transparent (alphaMode=blend) primitive over a WHITE
// background. A grazing silhouette hit crosses at most as much material as the
// body, so silhouette pixels must be no DARKER than the interior; anti-aliasing
// against the bright background can only brighten edges, so a silhouette that is
// darker than the interior is a genuine phantom-exit ring (a several-pixel band,
// well beyond the 1px AA rim). Averaged over the whole silhouette for robustness.
//
// NOTE: this does NOT gate the coincident-t dedup double-count at the EXACT
// tangent — that signal lives on the (measure-zero) silhouette which is also the
// anti-aliased edge, so it is not separable from AA at the render level. The
// dedup is gated deterministically at the solver level in
// test_IntersectorsOracle.cpp (testCoincidentDedup: a tangent grazes once, and a
// wall/cap rim crossing counts once, not twice).

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

static constexpr unsigned kRes = 256;
static constexpr int kSamples = 64;

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
  const char *subtype;
  vec3 p0, p1;
  float r0, r1;
};

static std::vector<vec4> render(anari::Device device, const Prim &prim)
{
  auto geometry = anari::newObject<anari::Geometry>(device, prim.subtype);
  if (prim.subtype[0] == 's') { // sphere
    anari::setParameterArray1D(device, geometry, "vertex.position", &prim.p0, 1);
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

  // Transparent, flat-appearing surface: blend a mid-gray over the background.
  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", vec3{0.15f, 0.15f, 0.15f});
  anari::setParameter(device, material, "opacity", 0.6f);
  anari::setParameter(device, material, "alphaMode", "blend");
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
  anari::setParameter(device, camera, "height", 2.4f);
  anari::commitParameters(device, camera);

  // White background, flat ambient so the surface reads as a near-constant color
  // (isolates transparency from shading gradients). No directional light.
  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "background", vec4{1.f, 1.f, 1.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 1.f);
  anari::setParameter(device, renderer, "pixelSamples", kSamples);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
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

static float luma(const vec4 &p)
{
  return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  int failures = 0;

  const std::array<Prim, 3> prims = {
      Prim{"sphere", {0.f, 0.f, 0.f}, {0, 0, 0}, 0.9f, 0.f},
      Prim{"cylinder", {-0.85f, 0.f, 0.f}, {0.85f, 0.f, 0.f}, 0.5f, 0.5f},
      Prim{"cone", {0.f, -0.85f, 0.f}, {0.f, 0.85f, 0.f}, 0.55f, 0.15f}};

  for (const auto &prim : prims) {
    const auto fb = render(device, prim);

    // Background is white (luma ~1). Covered = attenuated below the background.
    const auto isBg = [&](int x, int y) {
      return luma(fb[y * kRes + x]) > 0.9f;
    };
    const auto isCov = [&](int x, int y) { return !isBg(x, y); };

    // Any background within Chebyshev radius r of (x,y)?
    const auto bgWithin = [&](int x, int y, int r) {
      for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
          const int nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < int(kRes) && ny >= 0 && ny < int(kRes)
              && isBg(nx, ny))
            return true;
        }
      return false;
    };

    // Classify covered pixels by distance INTO the silhouette. A grazing
    // phantom exit over-attenuates a band a few pixels inside the edge (grazing
    // incidence lives just inside the silhouette), NOT the 1px AA rim. So skip
    // the outer 2px (AA), measure a RING 3-9px in, and compare it to the DEEP
    // interior (>=11px in) — a phantom ring is darker than the deep body.
    double ringSum = 0.0, deepSum = 0.0;
    int ringN = 0, deepN = 0;
    double ringMin = 1e30;
    bool nan = false;
    for (int y = 0; y < int(kRes); ++y) {
      for (int x = 0; x < int(kRes); ++x) {
        const vec4 &p = fb[y * kRes + x];
        for (int c = 0; c < 4; ++c)
          nan |= !std::isfinite(p[c]);
        if (!isCov(x, y) || bgWithin(x, y, 2))
          continue; // background or AA edge
        const double L = luma(p);
        // Ring spans 3..9px in (no gap up to the deep band) so a grazing
        // phantom, which lands just inside the silhouette, cannot slip between
        // the two regions.
        if (bgWithin(x, y, 9)) { // 3..9px inside — the phantom-exit band
          ringSum += L;
          ++ringN;
          ringMin = std::min(ringMin, L);
        } else if (!bgWithin(x, y, 10)) { // >=11px inside — deep body baseline
          deepSum += L;
          ++deepN;
        }
      }
    }

    if (nan) {
      fprintf(stderr, "FAIL: %s produced NaN/Inf\n", prim.subtype);
      ++failures;
    }
    if (deepN < 300 || ringN < 100) {
      fprintf(stderr, "FAIL: %s under-covered (deep=%d ring=%d)\n",
          prim.subtype, deepN, ringN);
      ++failures;
      continue;
    }
    const double deepMean = deepSum / deepN;
    const double ringMean = ringSum / ringN;

    if (getenv("XPARENCY_DEBUG"))
      fprintf(stderr, "%-9s deepMean=%.4f ringMean=%.4f ringMin=%.4f\n",
          prim.subtype, deepMean, ringMean, ringMin);

    // Sanity: the body is meaningfully attenuated below the white background.
    if (deepMean > 0.85) {
      fprintf(stderr, "FAIL: %s body not attenuated (mean=%.3f) — transparency "
                      "misconfigured\n",
          prim.subtype, deepMean);
      ++failures;
    }
    // The near-silhouette ring must be no darker than the deep body: a phantom
    // grazing exit adds a third attenuation there. (Both regions cross the same
    // two surfaces, so with correct gating ring == deep; if anything the ring is
    // slightly BRIGHTER as it approaches the single-surface silhouette.)
    if (ringMean < deepMean - 0.03) {
      fprintf(stderr,
          "FAIL: %s near-silhouette ring darker than body (ring=%.3f < "
          "deep=%.3f) — phantom grazing-exit dark ring\n",
          prim.subtype, ringMean, deepMean);
      ++failures;
    }
    if (ringMin < deepMean - 0.2) {
      fprintf(stderr,
          "FAIL: %s dark ring pixel (min=%.3f, deep=%.3f) — phantom grazing "
          "exit\n",
          prim.subtype, ringMin, deepMean);
      ++failures;
    }
  }

  anari::release(device, device);

  if (failures) {
    fprintf(stderr, "%d transparency phantom-exit failure(s)\n", failures);
    return 1;
  }
  printf("cone/cylinder/sphere transparency phantom-exit regression passed\n");
  return 0;
}
