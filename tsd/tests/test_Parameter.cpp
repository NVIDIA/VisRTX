// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/Parameter.hpp"

namespace {

struct MockObject : public tsd::core::ParameterObserver
{
  void parameterChanged(
      const tsd::core::Parameter *, const tsd::core::Any &) override
  {
    notified = true;
  }

  void removeParameter(const tsd::core::Parameter *) override
  {
    // no-op
  }

  bool notified{false};
};

} // namespace

///////////////////////////////////////////////////////////////////////////////

SCENARIO("tsd::core::Parameter interface", "[Parameter]")
{
  GIVEN("A constructed Parameter with a value")
  {
    MockObject obj;
    tsd::core::Parameter prop(&obj, "test_parameter");
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

  GIVEN("A constructed tsd::core::Parameter with a value and min/max")
  {
    MockObject obj;
    tsd::core::Parameter prop(&obj, "test_parameter");
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
