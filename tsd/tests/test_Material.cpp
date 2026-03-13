// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/scene/objects/Material.hpp"

using tsd::scene::Material;

SCENARIO("tsd::scene::Material interface", "[Material]")
{
  GIVEN("A default constructed Material")
  {
    Material obj;

    THEN("The object value type is correct")
    {
      REQUIRE(obj.type() == ANARI_MATERIAL);
    }
  }
}
