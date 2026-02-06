// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/Token.hpp"

SCENARIO("tsd::core::Token interface", "[Token]")
{
  GIVEN("A default constructed Token")
  {
    auto token = tsd::core::Token();

    THEN("The Token value is null")
    {
      REQUIRE(token.value() == nullptr);
    }

    THEN("A second default constructed Token will have the same value")
    {
      auto token2 = tsd::core::Token();
      REQUIRE(token == token2);
    }
  }

  GIVEN("A constructed Token from a given string")
  {
    auto token = tsd::core::Token("test1");

    THEN("The Token value is not null")
    {
      REQUIRE(token.value() != nullptr);
    }

    THEN("A second token constructed with the same string has the same value")
    {
      auto token2 = tsd::core::Token("test1");
      REQUIRE(token == token2);
    }

    THEN(
        "A second token constructed with a different string has the same "
        "value")
    {
      auto token2 = tsd::core::Token("test2");
      REQUIRE(token != token2);
    }
  }
}
