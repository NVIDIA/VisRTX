// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/AnyArray.hpp"
// anari
#include <anari/anari_cpp/ext/linalg.h>

SCENARIO("tsd::core::AnyArray interface", "[AnyArray]")
{
  GIVEN("A default constructed AnyArray")
  {
    tsd::core::AnyArray a;

    THEN("It is not valid and is empty")
    {
      REQUIRE(!a.valid());
      REQUIRE(!a);
      REQUIRE(a.empty());
      REQUIRE(a.size() == 0);
      REQUIRE(a.byteSize() == 0);
      REQUIRE(a.elementType() == ANARI_UNKNOWN);
    }
  }

  GIVEN("An AnyArray constructed with ANARI_FLOAT32 and count 4")
  {
    tsd::core::AnyArray a(ANARI_FLOAT32, 4);

    THEN("Metadata is correct")
    {
      REQUIRE(a.valid());
      REQUIRE(a);
      REQUIRE(!a.empty());
      REQUIRE(a.size() == 4);
      REQUIRE(a.elementSize() == sizeof(float));
      REQUIRE(a.byteSize() == 4 * sizeof(float));
      REQUIRE(a.elementType() == ANARI_FLOAT32);
      REQUIRE(a.is<float>());
      REQUIRE(a.is(ANARI_FLOAT32));
    }

    WHEN("Elements are set via get<float>(i) reference")
    {
      a.get<float>(0) = 1.f;
      a.get<float>(1) = 2.f;
      a.get<float>(2) = 3.f;
      a.get<float>(3) = 4.f;

      THEN("Elements can be read back correctly")
      {
        REQUIRE(a.get<float>(0) == 1.f);
        REQUIRE(a.get<float>(1) == 2.f);
        REQUIRE(a.get<float>(2) == 3.f);
        REQUIRE(a.get<float>(3) == 4.f);
      }

      THEN("dataAs<float>() gives the same values")
      {
        const float *p = a.dataAs<float>();
        REQUIRE(p[0] == 1.f);
        REQUIRE(p[1] == 2.f);
        REQUIRE(p[2] == 3.f);
        REQUIRE(p[3] == 4.f);
      }
    }

    WHEN("setElement(i, value) is used")
    {
      float v = 42.f;
      a.setElement(0, v);
      a.setElement<float>(1, 99.f);

      THEN("Elements are updated correctly")
      {
        REQUIRE(a.get<float>(0) == 42.f);
        REQUIRE(a.get<float>(1) == 99.f);
      }
    }

    WHEN("get<> is called with the wrong type")
    {
      THEN("It throws std::runtime_error")
      {
        REQUIRE_THROWS_AS(a.get<int>(0), std::runtime_error);
      }
    }
  }

  GIVEN("An AnyArray constructed from an ANARI_FLOAT32_VEC3 pointer")
  {
    using float3 = anari::math::float3;
    float3 src[2] = {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
    tsd::core::AnyArray a(ANARI_FLOAT32_VEC3, src, 2);

    THEN("Metadata is correct")
    {
      REQUIRE(a.size() == 2);
      REQUIRE(a.elementSize() == sizeof(float3));
      REQUIRE(a.byteSize() == 2 * sizeof(float3));
      REQUIRE(a.elementType() == ANARI_FLOAT32_VEC3);
    }

    THEN("Values are readable via dataAs<float3>()")
    {
      const float3 *p = a.dataAs<float3>();
      REQUIRE(p[0].x == 1.f);
      REQUIRE(p[0].y == 2.f);
      REQUIRE(p[0].z == 3.f);
      REQUIRE(p[1].x == 4.f);
      REQUIRE(p[1].y == 5.f);
      REQUIRE(p[1].z == 6.f);
    }

    THEN("Values are readable via get<float3>(i)")
    {
      REQUIRE(a.get<float3>(0).x == 1.f);
      REQUIRE(a.get<float3>(1).z == 6.f);
    }
  }

  GIVEN("An AnyArray of ANARI_FLOAT32 with 4 elements")
  {
    tsd::core::AnyArray a(ANARI_FLOAT32, 4);
    a.get<float>(0) = 10.f;
    a.get<float>(1) = 20.f;

    WHEN("It is copied")
    {
      tsd::core::AnyArray b = a;

      THEN("The copy has the same values")
      {
        REQUIRE(b.get<float>(0) == 10.f);
        REQUIRE(b.get<float>(1) == 20.f);
      }

      THEN("Modifying the copy does not affect the original")
      {
        b.get<float>(0) = 99.f;
        REQUIRE(a.get<float>(0) == 10.f);
      }
    }

    WHEN("resize() shrinks to 2 elements")
    {
      a.resize(2);

      THEN("Size and byteSize are updated")
      {
        REQUIRE(a.size() == 2);
        REQUIRE(a.byteSize() == 2 * sizeof(float));
      }
    }

    WHEN("reset() is called")
    {
      a.reset();

      THEN("The array becomes invalid and empty")
      {
        REQUIRE(!a.valid());
        REQUIRE(a.empty());
        REQUIRE(a.size() == 0);
        REQUIRE(a.elementType() == ANARI_UNKNOWN);
      }
    }
  }
}
