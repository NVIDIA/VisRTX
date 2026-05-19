// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/rendering/view/Manipulator.hpp"

namespace math = tsd::math;
namespace rendering = tsd::rendering;

static void requireNear(const math::float3 &a, const math::float3 &b)
{
  REQUIRE(math::neql(a.x, b.x, 1e-4f));
  REQUIRE(math::neql(a.y, b.y, 1e-4f));
  REQUIRE(math::neql(a.z, b.z, 1e-4f));
}

SCENARIO("Manipulator look mode preserves the camera anchor", "[Manipulator]")
{
  rendering::Manipulator m;
  m.setConfig(math::float3(0.f), 5.f, math::float2(30.f, 20.f));

  const auto orbitEye = m.eye();
  const auto orbitDir = m.dir();
  const auto orbitUp = m.up();

  WHEN("look mode is enabled")
  {
    m.setMode(rendering::ManipulatorMode::Look);

    THEN("the rendered camera pose does not jump")
    {
      requireNear(m.eye(), orbitEye);
      requireNear(m.dir(), orbitDir);
      requireNear(m.up(), orbitUp);
    }

    THEN("rotation keeps the camera position fixed")
    {
      m.startNewRotation();
      m.rotate(math::float2(0.1f, -0.05f));

      requireNear(m.eye(), orbitEye);
    }

    THEN("setting the center retargets without moving the camera")
    {
      m.setCenter(math::float3(1.f, 2.f, 3.f));

      requireNear(m.eye(), orbitEye);
      requireNear(m.at(), math::float3(1.f, 2.f, 3.f));
    }
  }
}
