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

// Stage 1.5 Geometry Light equivalence. An emissive sphere geometry
// (PhysicallyBased, constant `emissive`) must illuminate a floor the same as an
// ANARI sphere light (the "point" subtype with radius) of matching radiance,
// radius, and placement. Validates the sphere sampler's radiometry and that
// hit-side MIS does not bias (NEE + BSDF-hit deposit sum to the analytic light's
// contribution). The emitter sits above the top of frame so only its cast pool
// on the floor is measured — the emissive sphere's own glow (which the analytic
// light has no counterpart for) never enters the region. Rendered with 'quality'
// into a linear float buffer, firefly filter off.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
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
static constexpr int PIXEL_SAMPLES = 512;
static constexpr float EMISSIVE_RADIANCE = 8.f;

// Emitter footprint: a small sphere centered above the origin, above frame.
static constexpr vec3 SPHERE_CENTER = {0.f, 1.5f, 0.f};
static constexpr float SPHERE_RADIUS = 0.4f;

// Add a floor (large diffuse XZ quad at y=0) to the world.
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

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.6f, 0.6f, 0.6f});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// Emissive sphere geometry at the emitter footprint.
static anari::Surface makeEmissiveSphere(ANARIDevice device)
{
  const std::array<vec3, 1> pos = {SPHERE_CENTER};

  auto geom = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 1);
  anari::setParameter(device, geom, "radius", SPHERE_RADIUS);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  anari::setParameter(device,
      mat,
      "emissive",
      vec3{EMISSIVE_RADIANCE, EMISSIVE_RADIANCE, EMISSIVE_RADIANCE});
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// ANARI sphere light (the "point" subtype with radius) at the same footprint,
// radiance = intensity.
static anari::Light makeSphereLight(ANARIDevice device)
{
  auto light = anari::newObject<anari::Light>(device, "point");
  anari::setParameter(device, light, "color", vec3{1.f, 1.f, 1.f});
  anari::setParameter(device, light, "position", SPHERE_CENTER);
  anari::setParameter(device, light, "radius", SPHERE_RADIUS);
  anari::setParameter(device, light, "intensity", EMISSIVE_RADIANCE);
  anari::commitParameters(device, light);
  return light;
}

// Render the floor lit by either the emissive sphere (useMesh) or the ANARI
// sphere light, and return the linear framebuffer.
static std::vector<vec4> render(ANARIDevice device, bool useMesh)
{
  std::vector<anari::Surface> surfaces = {makeFloor(device)};
  std::vector<anari::Light> lights;
  if (useMesh)
    surfaces.push_back(makeEmissiveSphere(device));
  else
    lights.push_back(makeSphereLight(device));

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  if (!lights.empty())
    anari::setParameterArray1D(
        device, world, "light", lights.data(), lights.size());
  for (auto s : surfaces)
    anari::release(device, s);
  for (auto l : lights)
    anari::release(device, l);
  anari::commitParameters(device, world);

  // Look across the floor toward +Z; the emitter at y=1.5 sits above the frame,
  // so only its cast pool on the floor is visible.
  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.5f, -3.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, -0.15f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", PIXEL_SAMPLES);
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
  std::vector<vec4> out(fb.data, fb.data + size_t(fb.width) * fb.height);
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return out;
}

// Mean linear luminance over the lit floor region (lower-center of the frame,
// ANARI's bottom-left framebuffer origin).
static double poolMeanLuminance(const std::vector<vec4> &fb)
{
  double sum = 0.0;
  uint64_t n = 0;
  for (uint32_t y = IMAGE_SIZE[1] / 8; y < IMAGE_SIZE[1] / 2; ++y) {
    for (uint32_t x = 3 * IMAGE_SIZE[0] / 8; x < 5 * IMAGE_SIZE[0] / 8; ++x) {
      const vec4 &p = fb[y * IMAGE_SIZE[0] + x];
      sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
      ++n;
    }
  }
  return n ? sum / double(n) : 0.0;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const double mesh = poolMeanLuminance(render(device, true));
  const double light = poolMeanLuminance(render(device, false));

  anari::release(device, device);

  const double relErr =
      light > 0.0 ? std::abs(mesh - light) / light : (mesh > 0 ? 1.0 : 0.0);
  printf("emissiveSphere=%f  sphereLight=%f  relErr=%f\n", mesh, light, relErr);

  // Both estimators must converge to the same floor irradiance. A generous 5%
  // absorbs residual MC noise at 512 spp; a broken radiometry or double-count
  // shifts it by far more.
  if (mesh <= 0.0 || light <= 0.0) {
    fprintf(stderr, "FAIL: floor pool unlit (mesh=%f light=%f)\n", mesh, light);
    return 1;
  }
  constexpr double TOLERANCE = 0.05;
  if (relErr > TOLERANCE) {
    fprintf(stderr,
        "FAIL: emissive sphere geometry light not equivalent to sphere light (relErr=%f > %f)\n",
        relErr,
        TOLERANCE);
    return 1;
  }
  printf("emissive sphere geometry light equivalence passed\n");
  return 0;
}
