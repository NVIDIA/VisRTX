// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/Object.hpp"

namespace {

struct MockObject : public tsd::Object
{
  void parameterChanged(const tsd::Parameter *) override
  {
    notified = true;
  }

  bool notified{false};
};

} // namespace

SCENARIO("tsd::Object interface", "[Object]")
{
  GIVEN("A default constructed Object")
  {
    MockObject obj;

    THEN("The object value type is unknown")
    {
      REQUIRE(obj.type() == ANARI_UNKNOWN);
    }

    THEN("The object has no parameters")
    {
      REQUIRE(obj.numParameters() == 0);
    }

#if 0
    THEN("The object has no arrays")
    {
      REQUIRE(obj.numArrays() == 0);
    }
#endif

    WHEN("The object is given a parameter")
    {
      obj.setParameter("test", 5);

      THEN("The object has a single parameter")
      {
        REQUIRE(obj.numParameters() == 1);
      }

      THEN("The parameter is identical through index + token access")
      {
        REQUIRE(obj.parameter("test") == &obj.parameterAt(0));
      }

      THEN("The parameter value is correct")
      {
        auto &p = obj.parameterAt(0);
        REQUIRE(p.value().is<int>());
        REQUIRE(p.value().type() == ANARI_INT32);
        REQUIRE(p.value().get<int>() == 5);
      }

      THEN("Parameter notification should have occurred on initial set")
      {
        REQUIRE(obj.notified == true);
      }

      THEN("Changing the value of the parameter should cause notification")
      {
        obj.notified = false;
        obj.parameterAt(0) = 9;
        REQUIRE(obj.notified == true);
      }

      THEN("Removing the parameter results in no more parameters on the object")
      {
        obj.removeParameter("test");
        REQUIRE(obj.numParameters() == 0);
      }
    }
  }
}
