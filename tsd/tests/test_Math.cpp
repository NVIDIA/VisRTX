// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/TSDMath.hpp"
// std
#include <cmath>

namespace math = tsd::math;

static math::mat4 composeAzElRollTransform(const math::float3 &azelrot,
    const math::float3 &scale,
    const math::float3 &translation)
{
  auto rot = math::IDENTITY_MAT4;
  rot = math::mul(rot,
      math::rotation_matrix(math::rotation_quat(
          math::float3(0.f, 1.f, 0.f), math::radians(azelrot.x))));
  rot = math::mul(rot,
      math::rotation_matrix(math::rotation_quat(
          math::float3(1.f, 0.f, 0.f), math::radians(azelrot.y))));
  rot = math::mul(rot,
      math::rotation_matrix(math::rotation_quat(
          math::float3(0.f, 0.f, 1.f), math::radians(azelrot.z))));

  return math::mul(math::translation_matrix(translation),
      math::mul(rot, math::scaling_matrix(scale)));
}

static void requireMat4Near(
    const math::mat4 &actual, const math::mat4 &expected, float eps = 1e-4f)
{
  for (int c = 0; c < 4; c++) {
    for (int r = 0; r < 4; r++) {
      CAPTURE(c, r, actual[c][r], expected[c][r]);
      REQUIRE(std::abs(actual[c][r] - expected[c][r]) <= eps);
    }
  }
}

SCENARIO("Matrix decomposition test", "[Math]")
{
  GIVEN("Degree angles outside the UI range")
  {
    THEN("They normalize to the canonical 0 to 360 degree range")
    {
      REQUIRE(math::neql(math::normalizeDegrees(-90.f), 270.f));
      REQUIRE(math::neql(math::normalizeDegrees(450.f), 90.f));
      REQUIRE(math::normalizeDegrees(math::float3(-90.f, 360.f, 450.f))
          == math::float3(270.f, 0.f, 90.f));
    }
  }

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

SCENARIO("Matrix decomposition preserves singular SRT rotations", "[Math]")
{
  GIVEN("An SRT transform at the azimuth/elevation/roll singularity")
  {
    auto sc_in = math::float3(1.f, 1.f, 1.f);
    auto azelrot_in = math::float3(0.f, 90.f, 270.f);
    auto tl_in = math::float3(0.f, 0.f, 0.f);
    auto xfm = composeAzElRollTransform(azelrot_in, sc_in, tl_in);

    WHEN("The transform is decomposed to UI SRT and recomposed")
    {
      math::float3 sc_out, tl_out;
      math::mat4 rot_out;

      math::decomposeMatrix(xfm, sc_out, rot_out, tl_out);
      auto azelrot_out = math::degrees(math::matrixToAzElRoll(rot_out));
      auto roundtrip = composeAzElRollTransform(azelrot_out, sc_out, tl_out);

      THEN("The recomposed transform still matches the original")
      {
        requireMat4Near(roundtrip, xfm);
      }
    }
  }
}
