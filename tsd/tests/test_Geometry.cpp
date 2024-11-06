// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/objects/Geometry.hpp"

using tsd::Geometry;

SCENARIO("tsd::Geometry interface", "[Geometry]")
{
  GIVEN("A default constructed Geometry")
  {
    Geometry obj;

    THEN("The object value type is correct")
    {
      REQUIRE(obj.type() == ANARI_GEOMETRY);
    }
  }
}
