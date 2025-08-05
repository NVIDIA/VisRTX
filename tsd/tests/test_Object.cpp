// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/scene/Object.hpp"

namespace {

struct MockObject : public tsd::core::Object
{
  void parameterChanged(const tsd::core::Parameter *) override
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

    THEN("The object has no metadata")
    {
      REQUIRE(obj.numMetadata() == 0);
    }

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

    WHEN("An object metadata value is set")
    {
      obj.setMetadataValue("test_float", 5.f);

      THEN("The object now has 1 metadata on it")
      {
        REQUIRE(obj.numMetadata() == 1);
      }

      THEN("The set metadata name is correct")
      {
        REQUIRE(obj.getMetadataName(0) == std::string("test_float"));
      }

      THEN("The set metadata value is correct")
      {
        REQUIRE(obj.getMetadataValue("test_float").getAs<float>() == 5.f);
      }
    }

    WHEN("An object metadata array is set")
    {
      int arr[3] = {1, 2, 3};
      obj.setMetadataArray("test_array", ANARI_INT32, arr, 3);

      THEN("The object now has 1 metadata on it")
      {
        REQUIRE(obj.numMetadata() == 1);
      }

      THEN("The set metadata name is correct")
      {
        REQUIRE(obj.getMetadataName(0) == std::string("test_array"));
      }

      THEN("The set metadata array is correct")
      {
        const int *arr2 = nullptr;
        size_t size;
        anari::DataType type = ANARI_UNKNOWN;

        obj.getMetadataArray(
            "test_array", &type, (const void **)&arr2, &size);

        REQUIRE(type == ANARI_INT32);
        REQUIRE(size == 3);
        REQUIRE(arr2 != nullptr);
        REQUIRE(arr2[0] == 1);
        REQUIRE(arr2[1] == 2);
        REQUIRE(arr2[2] == 3);
      }
    }
  }
}
