// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/UpdateDelegate.hpp"
// std
#include <cmath>

namespace {

void requireMat4Near(const tsd::math::mat4 &actual,
    const tsd::math::mat4 &expected,
    float eps = 1e-4f)
{
  for (int c = 0; c < 4; c++) {
    for (int r = 0; r < 4; r++) {
      CAPTURE(c, r, actual[c][r], expected[c][r]);
      REQUIRE(std::abs(actual[c][r] - expected[c][r]) <= eps);
    }
  }
}

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

SCENARIO(
    "tsd::scene::LayerNodeData preserves singular SRT transforms", "[Scene]")
{
  GIVEN("A transform with elevation 90 and roll 270")
  {
    tsd::math::mat3 srt;
    srt[0] = tsd::math::float3(1.f, 1.f, 1.f);
    srt[1] = tsd::math::float3(0.f, 90.f, 270.f);
    srt[2] = tsd::math::float3(0.f, 0.f, 0.f);

    tsd::scene::LayerNodeData source(nullptr, srt);
    tsd::scene::LayerNodeData node(nullptr, source.getTransform());

    WHEN("The transform is exposed as UI SRT and applied back")
    {
      auto uiSrt = node.getTransformSRT();
      node.setAsTransform(uiSrt);

      THEN("The UI SRT keeps the roll and the matrix does not move")
      {
        REQUIRE(tsd::math::neql(uiSrt[1].x, 0.f, 1e-3f));
        REQUIRE(tsd::math::neql(uiSrt[1].y, 90.f, 1e-3f));
        REQUIRE(tsd::math::neql(uiSrt[1].z, 270.f, 1e-3f));
        requireMat4Near(node.getTransform(), source.getTransform());
      }
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
