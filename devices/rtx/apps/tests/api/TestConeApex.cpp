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

// GPU regression for issue #333 (cone rendering quality regression: the
// zero-radius apex frays / speckles). Renders a truncated cone tapering to a
// zero-radius apex side-on with an orthographic camera, so the silhouette is an
// exact triangle whose half-width at world height y is radius(y). Every pixel
// well inside that analytic silhouette must be covered by geometry; interior
// background pixels are apex fray/speckle. Asserts the interior is clean and
// that the apex region specifically renders (a solid taper, not a frayed tip).

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

// Cone geometry (object space): axis along +y, base radius rBase at y=yBase,
// zero-radius apex at y=yApex.
static constexpr float kYBase = -1.f;
static constexpr float kYApex = 1.f;
static constexpr float kRBase = 0.5f;
// Orthographic view frustum: square, world-height kViewH centered at the origin.
static constexpr float kViewH = 3.f;
static constexpr unsigned kRes = 256;

static float coneRadiusAt(float y)
{
  if (y < kYBase || y > kYApex)
    return -1.f;
  const float s = (y - kYBase) / (kYApex - kYBase); // 0 at base, 1 at apex
  return kRBase * (1.f - s);
}

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

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const uvec2 imageSize = {kRes, kRes};
  const vec4 background = {0.f, 0.f, 0.f, 1.f};

  auto geometry = anari::newObject<anari::Geometry>(device, "cone");
  {
    std::array<vec3, 2> positions = {
        vec3{0.f, kYBase, 0.f}, vec3{0.f, kYApex, 0.f}};
    std::array<float, 2> radii = {kRBase, 0.f};
    anari::setParameterArray1D(device,
        geometry,
        "vertex.position",
        positions.data(),
        positions.size());
    anari::setParameterArray1D(
        device, geometry, "vertex.radius", radii.data(), radii.size());
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

  // Orthographic camera looking along +z; the silhouette is then an exact
  // triangle in image space (half-width at height y == coneRadiusAt(y)).
  auto camera = anari::newObject<anari::Camera>(device, "orthographic");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -3.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(device, camera, "aspect", 1.f);
  anari::setParameter(device, camera, "height", kViewH);
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

  const float pixel = kViewH / float(kRes); // world units per pixel
  const auto isGeom = [&](unsigned px, unsigned py) {
    const uint32_t c = fb.data[py * fb.width + px];
    return (c & 0xFFu) > 16u; // red channel (RGBA8, little-endian)
  };

  // Scan the analytic interior: shrink the silhouette by a margin so anti-
  // aliased edge pixels are excluded, and only test where the cone is at least
  // a couple pixels wide (nearer the apex than that is legitimately sub-pixel).
  int interiorHoles = 0;
  int apexInteriorTested = 0; // interior samples in the top 25% toward the apex
  int apexInteriorHit = 0;
  const float yApexZoneStart = kYApex - 0.25f * (kYApex - kYBase);
  for (unsigned py = 0; py < kRes; ++py) {
    // ANARI's color buffer origin is bottom-left: row 0 is the bottom (-y).
    const float wy = ((py + 0.5f) / kRes - 0.5f) * kViewH;
    const float r = coneRadiusAt(wy);
    if (r < 2.f * pixel)
      continue; // outside the cone or sub-2-pixel taper tip
    const float rInner = r - 1.5f * pixel; // strip AA edge
    if (rInner <= 0.f)
      continue;
    for (unsigned px = 0; px < kRes; ++px) {
      const float wx = ((px + 0.5f) / kRes - 0.5f) * kViewH;
      if (wx < -rInner || wx > rInner)
        continue; // outside the shrunk silhouette
      const bool hit = isGeom(px, py);
      if (!hit)
        ++interiorHoles;
      if (wy >= yApexZoneStart) {
        ++apexInteriorTested;
        apexInteriorHit += hit ? 1 : 0;
      }
    }
  }

  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  anari::release(device, device);

  int failures = 0;
  // The interior of a convex cone silhouette must be watertight — any hole is
  // apex fray/speckle (issue #333).
  if (interiorHoles > 0) {
    fprintf(stderr,
        "FAIL: %d interior background pixel(s) — cone apex fray/speckle "
        "(#333 regression)\n",
        interiorHoles);
    ++failures;
  }
  // The apex zone must actually render (guards against the taper vanishing
  // entirely rather than fraying).
  if (apexInteriorTested == 0) {
    fprintf(stderr, "FAIL: apex zone produced no interior test samples\n");
    ++failures;
  } else if (apexInteriorHit < apexInteriorTested) {
    fprintf(stderr,
        "FAIL: apex zone %d/%d interior samples covered (taper frays toward "
        "the apex)\n",
        apexInteriorHit,
        apexInteriorTested);
    ++failures;
  }

  if (failures)
    return 1;
  printf("cone apex #333 regression passed (%d apex-zone interior samples "
         "clean)\n",
      apexInteriorTested);
  return 0;
}
