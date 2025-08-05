// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/scene/objects/Material.hpp"

using tsd::core::Material;

SCENARIO("tsd::core::Material interface", "[Material]")
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
