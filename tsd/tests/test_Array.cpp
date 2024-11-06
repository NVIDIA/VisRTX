// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/objects/Array.hpp"

using namespace tsd::literals;

SCENARIO("tsd::Array interface", "[Array]")
{
  GIVEN("A default constructed Array")
  {
    tsd::Array arr;

    THEN("The array is empty")
    {
      REQUIRE(arr.size() == 0);
    }

    THEN("The array element type is unknown")
    {
      REQUIRE(arr.elementType() == ANARI_UNKNOWN);
    }
  }

  GIVEN("A constructed int Array")
  {
    auto arr = tsd::Array(ANARI_INT32, 4);

    THEN("The array is not empty")
    {
      REQUIRE(arr.size() == 4);
    }

    THEN("The array shape is correct")
    {
      REQUIRE(arr.shape() == 1);
    }

    THEN("The array element type is correct")
    {
      REQUIRE(arr.elementType() == ANARI_INT32);
    }

    THEN("Mapping the array is not null")
    {
      void *m = arr.map();
      REQUIRE(m != nullptr);
      arr.unmap();
    }
  }
}
