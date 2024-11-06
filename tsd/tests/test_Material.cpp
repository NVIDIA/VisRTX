// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/objects/Material.hpp"

using tsd::Material;

SCENARIO("tsd::Material interface", "[Material]")
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
