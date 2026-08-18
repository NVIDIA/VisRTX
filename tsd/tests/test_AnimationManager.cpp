// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/UpdateDelegate.hpp"
#include "tsd/scene/objects/Array.hpp"
// std
#include <vector>

using tsd::animation::AnimationManager;
using tsd::scene::Scene;

namespace {

// Records the batch bracket and the array unmaps inside it, which is the
// signal a render index coalesces its world rebuilds on.
struct BatchRecordingDelegate : public tsd::scene::EmptyUpdateDelegate
{
  void signalUpdateBatchBegin() override
  {
    depth++;
    begins++;
  }

  void signalUpdateBatchEnd() override
  {
    depth--;
    ends++;
  }

  void signalArrayUnmapped(const tsd::scene::Array *) override
  {
    unmapsInsideBatch += depth > 0 ? 1 : 0;
    unmapsOutsideBatch += depth > 0 ? 0 : 1;
  }

  int depth{0};
  int begins{0};
  int ends{0};
  int unmapsInsideBatch{0};
  int unmapsOutsideBatch{0};
};

} // namespace

SCENARIO("A time change is one update batch", "[AnimationManager]")
{
  GIVEN("Several bindings that each rewrite an Array")
  {
    Scene scene;
    AnimationManager mgr(&scene);

    auto *recorder =
        scene.updateDelegate().emplace<BatchRecordingDelegate>();

    std::vector<tsd::scene::ArrayRef> arrays;
    for (int i = 0; i < 3; ++i) {
      auto &anim = mgr.addAnimation("rewriter" + std::to_string(i));
      auto array = scene.createArray(ANARI_FLOAT32_MAT4, 1);
      arrays.push_back(array);
      anim.addCallbackBinding([array](float t) mutable {
        const auto m = tsd::math::IDENTITY_MAT4;
        array->setData(&m, 1);
      });
    }

    WHEN("The animation time changes once")
    {
      const int unmapsBefore = recorder->unmapsOutsideBatch;
      mgr.setAnimationTime(0.5f);

      THEN("Every rewrite lands inside exactly one balanced batch")
      {
        REQUIRE(recorder->begins == 1);
        REQUIRE(recorder->ends == 1);
        REQUIRE(recorder->depth == 0);
        REQUIRE(recorder->unmapsInsideBatch == 3);
        REQUIRE(recorder->unmapsOutsideBatch == unmapsBefore);
      }
    }
  }
}

SCENARIO("tsd::animation::AnimationManager playback", "[AnimationManager]")
{
  Scene scene;
  AnimationManager mgr(&scene);

  GIVEN("A manager configured for deterministic frame playback")
  {
    mgr.setAnimationTotalFrames(5);
    mgr.setAnimationFPS(2.f);

    WHEN("Playing with one frame worth of elapsed wall-clock time")
    {
      mgr.play();
      mgr.tick(0.5f);

      THEN("Playback advances exactly one frame")
      {
        REQUIRE(mgr.getAnimationFrame() == 1);
      }
    }

    WHEN("A slow frame accumulates enough time for multiple animation steps")
    {
      mgr.play();
      mgr.tick(1.25f);

      THEN("Playback catches up by advancing multiple frames")
      {
        REQUIRE(mgr.getAnimationFrame() == 2);
      }
    }

    WHEN("Looping playback advances past the last frame")
    {
      mgr.setAnimationFrame(4);
      mgr.play();
      mgr.tick(0.5f);

      THEN("Playback wraps back to the first frame")
      {
        REQUIRE(mgr.getAnimationFrame() == 0);
      }
    }

    WHEN("Non-looping playback reaches the last frame")
    {
      mgr.setLoop(false);
      mgr.setAnimationFrame(3);
      mgr.play();
      mgr.tick(1.0f);

      THEN("Playback stops on the last frame")
      {
        REQUIRE(mgr.getAnimationFrame() == 4);
        REQUIRE_FALSE(mgr.isPlaying());
      }
    }

    WHEN("An explicit seek happens after partial playback accumulation")
    {
      mgr.play();
      mgr.tick(0.25f);
      mgr.setAnimationFrame(2);
      mgr.tick(0.25f);

      THEN("The seek clears the accumulator so no extra frame is consumed")
      {
        REQUIRE(mgr.getAnimationFrame() == 2);
      }
    }
  }

  GIVEN("An animation manager with custom playback settings")
  {
    mgr.setAnimationTime(0.3f);
    mgr.setAnimationIncrement(0.2f);
    mgr.setAnimationTotalFrames(9);
    mgr.setAnimationFPS(12.f);

    WHEN("The manager is serialized and restored")
    {
      tsd::core::DataTree tree;
      tsd::io::animationManagerToNode(mgr, tree.root()["animations"]);

      Scene restoredScene;
      AnimationManager restored(&restoredScene);
      tsd::io::nodeToAnimationManager(
          tree.root()["animations"], restored, restoredScene);

      THEN("Playback FPS and existing timing state round-trip")
      {
        REQUIRE(restored.getAnimationTime() == Approx(0.3f));
        REQUIRE(restored.getAnimationIncrement() == Approx(0.2f));
        REQUIRE(restored.getAnimationTotalFrames() == 9);
        REQUIRE(restored.getAnimationFPS() == Approx(12.f));
      }
    }
  }
}
