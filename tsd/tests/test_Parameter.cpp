// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/scene/Parameter.hpp"
// std
#include <string>

namespace {

struct MockObject : public tsd::scene::ParameterObserver
{
  void parameterChanged(
      const tsd::scene::Parameter *, const tsd::core::Any &) override
  {
    notified = true;
  }

  void removeParameter(const tsd::scene::Parameter *) override
  {
    // no-op
  }

  bool notified{false};
};

} // namespace

///////////////////////////////////////////////////////////////////////////////

SCENARIO("tsd::scene::Parameter interface", "[Parameter]")
{
  GIVEN("A constructed Parameter with a value")
  {
    MockObject obj;
    tsd::scene::Parameter prop(&obj, "test_parameter");
    prop.setValue(5);

    THEN("The Parameter name is correct")
    {
      REQUIRE(prop.name() == "test_parameter");
    }

    THEN("The Parameter value type is correct")
    {
      REQUIRE(prop.value().is<int>());
      REQUIRE(!prop.value().is<float>());
    }

    THEN("The Parameter value is correct")
    {
      REQUIRE(prop.value().get<int>() == 5);
    }

    THEN("The Parameter type is correct")
    {
      REQUIRE(prop.value().type() == ANARI_INT32);
    }

    THEN("The Parameter has no min value")
    {
      REQUIRE(!prop.hasMin());
    }

    THEN("The Parameter has no max value")
    {
      REQUIRE(!prop.hasMax());
    }

    THEN("The Parameter's object observing it has been notified already")
    {
      REQUIRE(obj.notified);
    }

    THEN("Setting the Parameter's value changes the value and notifies")
    {
      prop.setValue(10);

      REQUIRE(prop.value().is<int>());
      REQUIRE(prop.value().get<int>() == 10);
      REQUIRE(obj.notified);
    }
  }

  GIVEN("A constructed tsd::scene::Parameter with a value and min/max")
  {
    MockObject obj;
    tsd::scene::Parameter prop(&obj, "test_parameter");
    prop.setValue(5)
        .setDescription("this is a test parameter")
        .setMin(0)
        .setMax(10);

    THEN("The Parameter has a min value")
    {
      REQUIRE(prop.hasMin());
      REQUIRE(prop.min().get<int>() == 0);
    }

    THEN("The Parameter has no max value")
    {
      REQUIRE(prop.hasMax());
      REQUIRE(prop.max().get<int>() == 10);
    }

    THEN("The Parameter min value can be reset through setMin({})")
    {
      prop.setMin({});
      REQUIRE(!prop.hasMin());
    }

    THEN("The Parameter max value can be reset through setMax({})")
    {
      prop.setMax({});
      REQUIRE(!prop.hasMax());
    }
  }
}

SCENARIO("tsd::core::Any recognizes strings through the generic accessors",
    "[Parameter]")
{
  // ANARITypeFor<std::string> is ANARI_UNKNOWN, so the generic is<>/get<>
  // templates silently miss ANARI_STRING unless Any specializes them. Callers
  // that reach for is<std::string>() -- the Lua bindings among them -- then
  // treat every string parameter as absent.
  GIVEN("An Any holding a string")
  {
    tsd::core::Any value = std::string("hello");

    THEN("Its type is ANARI_STRING")
    {
      REQUIRE(value.type() == ANARI_STRING);
    }

    THEN("is<std::string>() reports the string")
    {
      REQUIRE(value.is<std::string>());
      REQUIRE(!value.is<int>());
      REQUIRE(!value.is<float>());
    }

    THEN("get<std::string>() returns the string")
    {
      REQUIRE(value.get<std::string>() == "hello");
    }

    THEN("getValueOr<std::string>() returns the string")
    {
      REQUIRE(value.getValueOr<std::string>("fallback") == "hello");
    }
  }

  GIVEN("An Any holding a non-string")
  {
    tsd::core::Any value = 5;

    THEN("It is not mistaken for a string")
    {
      REQUIRE(!value.is<std::string>());
      REQUIRE(value.getValueOr<std::string>("fallback") == "fallback");
    }
  }
}
