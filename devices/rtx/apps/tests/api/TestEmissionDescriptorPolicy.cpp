/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Device-level checks of the emission descriptor registration policy (ADR
// 0007): a raw `mdl` surface becomes a next-event-sampled Geometry Light iff
// its folded descriptor is non-null AND faithfully NEE-evaluable — diffuse EDF,
// radiant- exitance mode, provably non-negative, no geometric-state dependence.
// Each case is asserted via the world's `numLightInstances` property (a
// zero-radiance or forward-only emitter is pixel-identical to no light, so the
// count is the only observable seam). Emission excluded here still renders via
// the forward path, unbiased — this test asserts registration, not radiance.

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

using vec3 = std::array<float, 3>;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_WARNING)
    fprintf(stderr, "[anari] %s\n", message);
}

// A control: constant diffuse radiant-exitance emission — registers.
static const char *POLICY_DIFFUSE = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
export material diffuse_emit() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color(8.0) * math::PI)));
)mdl";

// Non-diffuse EDF: not in faithfulSet — described, not registered.
static const char *POLICY_SPOT = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
export material spot_emit() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::spot_edf(exponent: 1.0),
            intensity: color(8.0) * math::PI)));
)mdl";

// Provably-negative emission: sign gate excludes it (its all-negative
// next-event contribution would be dropped by the shadow epsilon gate while the
// forward deposit is MIS-downweighted — bias). Forward-only, unbiased.
static const char *POLICY_NEGATIVE = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
export material negative_emit() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color(-8.0) * math::PI)));
)mdl";

// Intensity reads state::normal — a geometric-state quantity the synthetic hit
// fabricates. Faithful EDF kind, but unfaithful integrand: not registered.
static const char *POLICY_STATE = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
import ::state::*;
export material state_emit() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color(math::length(state::normal())) * math::PI)));
)mdl";

// Power intensity mode: not faithfully handled — described, not registered.
static const char *POLICY_POWER = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
export material power_emit() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color(8.0) * math::PI,
            mode: intensity_power)));
)mdl";

static bool checkLightCount(
    ANARIDevice device, const char *source, const char *name, uint32_t expected)
{
  const std::array<vec3, 4> pos = {vec3{-0.5f, 1.5f, -0.5f},
      vec3{0.5f, 1.5f, -0.5f},
      vec3{0.5f, 1.5f, 0.5f},
      vec3{-0.5f, 1.5f, 0.5f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "mdl");
  anari::setParameter(device, mat, "sourceType", "code");
  anari::setParameter(device, mat, "source", source);
  anari::setParameter(device, mat, "materialName", name);
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  uint32_t count = ~0u;
  const bool found =
      anari::getProperty(device, world, "numLightInstances", count, ANARI_WAIT);
  anari::release(device, world);

  printf("numLightInstances(%s)=%u (expected %u)\n", name, count, expected);
  if (!found) {
    fprintf(stderr, "FAIL: world has no numLightInstances property\n");
    return false;
  }
  if (count != expected) {
    fprintf(stderr,
        "FAIL: %s expected %u light(s), got %u\n",
        name,
        expected,
        count);
    return false;
  }
  return true;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  if (!device) {
    fprintf(stderr, "FAIL: could not create VisRTX device\n");
    return 1;
  }

  bool ok = true;
  // Control: a faithful diffuse emitter registers.
  ok = checkLightCount(device, POLICY_DIFFUSE, "diffuse_emit", 1) && ok;
  // Each faithfulness gate excludes its case (described, forward-only).
  ok = checkLightCount(device, POLICY_SPOT, "spot_emit", 0) && ok;
  ok = checkLightCount(device, POLICY_NEGATIVE, "negative_emit", 0) && ok;
  ok = checkLightCount(device, POLICY_STATE, "state_emit", 0) && ok;
  ok = checkLightCount(device, POLICY_POWER, "power_emit", 0) && ok;

  anari::release(device, device);

  if (!ok) {
    fprintf(stderr, "TestEmissionDescriptorPolicy FAILED\n");
    return 1;
  }
  printf("TestEmissionDescriptorPolicy passed\n");
  return 0;
}
