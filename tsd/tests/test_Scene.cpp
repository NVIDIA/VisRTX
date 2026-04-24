// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/UpdateDelegate.hpp"

namespace {

struct CountingDelegate : public tsd::scene::EmptyUpdateDelegate
{
  CountingDelegate(int *objectAddedCount) : m_objectAddedCount(objectAddedCount)
  {}

  void signalObjectAdded(const tsd::scene::Object *) override
  {
    if (m_objectAddedCount)
      (*m_objectAddedCount)++;
  }

  int *m_objectAddedCount{nullptr};
};

} // namespace

SCENARIO("tsd::scene::Scene owns an intrinsic update delegate root", "[Scene]")
{
  GIVEN("A scene")
  {
    tsd::scene::Scene scene;
    auto &delegate = scene.updateDelegate();

    THEN("The scene starts with an empty MultiUpdateDelegate root")
    {
      REQUIRE(delegate.size() == 0);
    }

    THEN("Const and non-const accessors return the same delegate root")
    {
      const auto &constScene = scene;
      REQUIRE(&constScene.updateDelegate() == &delegate);
    }

    THEN("Registering a child delegate observes new scene-owned objects and arrays")
    {
      int objectAddedCount = 0;
      auto *countingDelegate = scene.updateDelegate().emplace<CountingDelegate>(
          &objectAddedCount);
      auto geometry = scene.createObject<tsd::scene::Geometry>(
          tsd::scene::tokens::geometry::sphere);
      auto array = scene.createArray(ANARI_FLOAT32, 4);

      REQUIRE(geometry);
      REQUIRE(array);
      REQUIRE(countingDelegate != nullptr);
      REQUIRE(objectAddedCount == 2);
    }
  }
}

SCENARIO("tsd::scene::Scene delegate registration controls live signaling",
    "[Scene]")
{
  GIVEN("A scene with an explicitly registered child delegate")
  {
    tsd::scene::Scene scene;
    int objectAddedCount = 0;
    auto *delegate = scene.updateDelegate().emplace<CountingDelegate>(
        &objectAddedCount);

    REQUIRE(scene.updateDelegate().size() == 1);

    WHEN("A new scene object is created while the delegate is registered")
    {
      auto geometry = scene.createObject<tsd::scene::Geometry>(
          tsd::scene::tokens::geometry::sphere);

      THEN("The delegate receives the object-added signal")
      {
        REQUIRE(geometry);
        REQUIRE(objectAddedCount == 1);
      }
    }

    WHEN("The delegate is erased before more scene changes occur")
    {
      scene.updateDelegate().erase(delegate);
      auto geometry = scene.createObject<tsd::scene::Geometry>(
          tsd::scene::tokens::geometry::sphere);

      THEN("The root becomes empty and no further signals are delivered")
      {
        REQUIRE(geometry);
        REQUIRE(scene.updateDelegate().size() == 0);
        REQUIRE(objectAddedCount == 0);
      }
    }
  }
}
