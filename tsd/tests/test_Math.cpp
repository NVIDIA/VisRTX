// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/TSDMath.hpp"

namespace math = tsd::math;

SCENARIO("Matrix decomposition test", "[Math]")
{
  GIVEN("An SRT formulated matrix transform")
  {
    auto tl_in = math::float3(1.f, 2.f, 3.f);
    auto rot_in = math::rotation_matrix(
        math::rotation_quat(math::float3(0.f, 1.f, 0.f), math::radians(45.f)));
    auto sc_in = math::float3(2.f, 3.f, 4.f);

    math::mat4 xfm = math::mul(math::translation_matrix(tl_in),
        math::mul(rot_in, math::scaling_matrix(sc_in)));

    WHEN("The transform is decomposed")
    {
      math::float3 sc_out, tl_out;
      math::mat4 rot_out;

      math::decomposeMatrix(xfm, sc_out, rot_out, tl_out);

      THEN("The scale is correct")
      {
        REQUIRE(sc_in == sc_out);
      }

      THEN("The rotation is correct")
      {
        REQUIRE(rot_in == rot_out);
      }

      THEN("The translation is correct")
      {
        REQUIRE(tl_in == tl_out);
      }

      THEN("The az-el-roll of the rotation is correct")
      {
        auto azelro = math::matrixToAzElRoll(rot_out);
        REQUIRE(math::neql(math::degrees(azelro.x), 45.f, 1e-3f));
        REQUIRE(math::neql(math::degrees(azelro.y), 0.f, 1e-3f));
        REQUIRE(math::neql(math::degrees(azelro.z), 0.f, 1e-3f));
      }
    }
  }
}
