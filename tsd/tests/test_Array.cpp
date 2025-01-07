// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef TSD_USE_CUDA
#undef TSD_USE_CUDA
#define TSD_USE_CUDA 0
#endif

// catch
#include "catch.hpp"
// tsd
#include "tsd/algorithms/computeScalarRange.hpp"
#include "tsd/objects/Array.hpp"
// std
#include <numeric>

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

    THEN("The array type/shape is correct")
    {
      REQUIRE(arr.type() == ANARI_ARRAY1D);
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

  GIVEN("A constructed float Array with linear elements")
  {
    auto arr = tsd::Array(ANARI_UFIXED8, 256);
    auto *begin = arr.mapAs<uint8_t>();
    auto *end = begin + 256;
    std::iota(begin, end, 0);
    arr.unmap();

    THEN("Computing the range will return the correct min + max")
    {
      auto range = tsd::algorithm::computeScalarRange(arr);
      REQUIRE(range.x == 0.f);
      REQUIRE(range.y == 1.f);
    }
  }
}
