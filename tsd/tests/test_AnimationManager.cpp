// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/io/serialization.hpp"
#include "tsd/scene/Scene.hpp"

using tsd::animation::AnimationManager;
using tsd::scene::Scene;

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
