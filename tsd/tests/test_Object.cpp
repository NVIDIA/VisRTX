// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/scene/Object.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <memory>

namespace {

struct MockObject : public tsd::scene::Object
{
  void parameterChanged(
      const tsd::scene::Parameter *, const tsd::core::Any &) override
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

        obj.getMetadataArray("test_array", &type, (const void **)&arr2, &size);

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

SCENARIO("tsd::Object clone for scene objects", "[Object]")
{
  auto scene = std::make_unique<tsd::scene::Scene>();

  GIVEN("A scene-owned surface with parameters and metadata")
  {
    auto geometry = scene->createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::sphere);
    auto material = scene->createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    auto surface = scene->createSurface("primary_surface", geometry, material);
    surface->setParameter("testFloat", 3.5f);
    surface->setMetadataValue("priority", 7);
    const int metadataArray[3] = {1, 2, 3};
    surface->setMetadataArray("bins", ANARI_INT32, metadataArray, 3);

    auto *clone = tsd::scene::cloneObject(surface.data());

    THEN(
        "The clone preserves type, subtype, references, parameters, and metadata")
    {
      REQUIRE(clone != nullptr);
      REQUIRE(clone != surface.data());
      REQUIRE(clone->type() == surface->type());
      REQUIRE(clone->subtype() == surface->subtype());
      REQUIRE(clone->name() == "primary_surface_clone");
      REQUIRE(clone->numParameters() == surface->numParameters());
      REQUIRE(clone->numMetadata() == surface->numMetadata());
      REQUIRE(
          clone->parameterValueAs<float>("testFloat").value() == Approx(3.5f));
      REQUIRE(clone->getMetadataValue("priority").getAs<int>() == 7);

      auto *cloneSurface = dynamic_cast<tsd::scene::Surface *>(clone);
      REQUIRE(cloneSurface != nullptr);
      REQUIRE(cloneSurface->geometry() == geometry.data());
      REQUIRE(cloneSurface->material() == material.data());

      anari::DataType type = ANARI_UNKNOWN;
      const int *values = nullptr;
      size_t size = 0;
      clone->getMetadataArray("bins", &type, (const void **)&values, &size);
      REQUIRE(type == ANARI_INT32);
      REQUIRE(size == 3);
      REQUIRE(values[0] == 1);
      REQUIRE(values[1] == 2);
      REQUIRE(values[2] == 3);
    }

    THEN("Changing the clone does not mutate the original")
    {
      REQUIRE(clone != nullptr);
      clone->setParameter("testFloat", 9.f);
      clone->setMetadataValue("priority", 99);

      REQUIRE(
          clone->parameterValueAs<float>("testFloat").value() == Approx(9.f));
      REQUIRE(surface->parameterValueAs<float>("testFloat").value()
          == Approx(3.5f));
      REQUIRE(clone->getMetadataValue("priority").getAs<int>() == 99);
      REQUIRE(surface->getMetadataValue("priority").getAs<int>() == 7);
    }
  }

  GIVEN("A scene-owned renderer with a device name")
  {
    auto renderer =
        scene->createRenderer("visrtx", tsd::scene::tokens::defaultToken);
    renderer->setName("main_renderer");
    renderer->setParameter("ambientRadiance", 1.25f);
    renderer->setMetadataValue("quality", 4);

    auto *clone = tsd::scene::cloneObject(renderer.get());

    THEN("The clone preserves renderer-specific state")
    {
      auto *rendererClone = dynamic_cast<tsd::scene::Renderer *>(clone);
      REQUIRE(rendererClone != nullptr);
      REQUIRE(rendererClone->rendererDeviceName() == "visrtx");
      REQUIRE(rendererClone->subtype() == renderer->subtype());
      REQUIRE(rendererClone->name() == "main_renderer_clone");
      REQUIRE(rendererClone->parameterValueAs<float>("ambientRadiance").value()
          == Approx(1.25f));
      REQUIRE(rendererClone->getMetadataValue("quality").getAs<int>() == 4);
    }
  }
}
