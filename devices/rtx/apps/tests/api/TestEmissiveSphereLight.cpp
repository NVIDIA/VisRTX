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
// hit-side MIS does not bias (NEE + BSDF-hit deposit sum to the analytic
// light's contribution). The emitter sits above the top of frame so only its
// cast pool on the floor is measured — the emissive sphere's own glow (which
// the analytic light has no counterpart for) never enters the region. Rendered
// with 'quality' into a linear float buffer, firefly filter off.

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
#include "DeviceTestArray.h"
#include "ReceiverTestSupport.h"

struct ResidencyCase
{
  const char *name;
  int deviceInputs;
};

// Bounds/visibility isolate input roles; radiometry checks the sampler.
static constexpr std::array<ResidencyCase, 2> SPHERE_CASES = {
    {{"host-indexed", 0}, {"all-device", 7}}};

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static constexpr uvec2 IMAGE_SIZE = {256, 256};
static constexpr int PIXEL_SAMPLES = 512;
static constexpr float EMISSIVE_RADIANCE = 8.f;
static constexpr double TOLERANCE = 0.05;

static bool equivalent(double actual, double reference)
{
  return std::isfinite(actual) && std::isfinite(reference) && actual > 0.0
      && reference > 0.0
      && std::abs(actual - reference) / reference <= TOLERANCE;
}

// Emitter footprint: a small sphere centered above the origin, above frame.
static constexpr vec3 SPHERE_CENTER = {0.f, 1.5f, 0.f};
static constexpr float SPHERE_RADIUS = 0.4f;

// Emissive sphere geometry at the emitter footprint.
static anari::Surface makeEmissiveSphere(
    ANARIDevice device, int deviceInputs, bool indexed)
{
  const std::array<vec3, 1> pos = {SPHERE_CENTER};

  auto geom = anari::newObject<anari::Geometry>(device, "sphere");
  if (indexed) {
    const std::array<vec3, 2> positions = {vec3{0.f, 20.f, 0.f}, SPHERE_CENTER};
    const std::array<float, 2> radii = {0.01f, SPHERE_RADIUS};
    const std::array<unsigned, 1> indices = {1};
    setTestArray(device, geom, "vertex.position", positions, deviceInputs & 1);
    setTestArray(device, geom, "vertex.radius", radii, deviceInputs & 2);
    setTestArray(device, geom, "primitive.index", indices, deviceInputs & 4);
    anari::setParameter(device, geom, "radius", 0.01f);
  } else {
    anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 1);
    anari::setParameter(device, geom, "radius", SPHERE_RADIUS);
  }
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
// sphere light, measuring linear luminance without copying the framebuffer.
static double render(ANARIDevice device,
    bool useMesh,
    int deviceInputs = 0,
    bool indexed = false)
{
  std::vector<anari::Surface> surfaces = {visrtx::test::receiverFloor(device)};
  std::vector<anari::Light> lights;
  if (useMesh)
    surfaces.push_back(makeEmissiveSphere(device, deviceInputs, indexed));
  else
    lights.push_back(makeSphereLight(device));

  auto world = anari::newObject<anari::World>(device);
  const visrtx::test::ObjectOwner worldOwner(device, world);
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

  auto frame = visrtx::test::receiverFrame(
      device, world, IMAGE_SIZE, "quality", PIXEL_SAMPLES);
  const visrtx::test::ObjectOwner frameOwner(device, frame);
  // Bottom-left origin; only the floor's cast pool, never the emitter.
  return visrtx::test::receiverLuminance(device,
      frame,
      IMAGE_SIZE,
      {3 * IMAGE_SIZE[0] / 8,
          5 * IMAGE_SIZE[0] / 8,
          IMAGE_SIZE[1] / 8,
          IMAGE_SIZE[1] / 2});
}

int main()
{
  auto device = makeVisRTXDevice(visrtx::test::exitOnErrorStatus);

  fprintf(stderr, "[case] sphere/reference\n");
  const double light = render(device, false);
  bool ok = std::isfinite(light) && light > 0.0;

  const auto check = [&](const char *name, int deviceInputs, bool indexed) {
    fprintf(stderr, "[case] sphere/%s\n", name);
    const double mesh = render(device, true, deviceInputs, indexed);
    // The unchanged 5% allowance covers MC noise at 512 spp.
    const bool matches = equivalent(mesh, light);
    printf("sphere/%s mesh=%f reference=%f relErr=%g tolerance=%g %s\n",
        name,
        mesh,
        light,
        std::abs(mesh - light) / light,
        TOLERANCE,
        matches ? "PASS" : "FAIL");
    ok = matches && ok;
  };
  check("host-scalar-radius", 0, false);
  for (const auto &residency : SPHERE_CASES)
    check(residency.name, residency.deviceInputs, true);

  anari::release(device, device);
  return ok ? 0 : 1;
}
