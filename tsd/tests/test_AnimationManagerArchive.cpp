// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/io/archives/AnimationManagerArchive.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <cstdio>
#include <filesystem>

SCENARIO(
    "Animation Manager Archives replace validated state and restore stopped",
    "[AnimationManagerArchive]")
{
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager source(&scene);
  auto geometry = scene.createObject<tsd::scene::Geometry>("sphere");
  source.setAnimationTime(0.25f);
  source.setAnimationIncrement(0.125f);
  source.setAnimationTotalFrames(41);
  source.setAnimationFPS(24.f);
  source.setLoop(false);
  source.play();
  const float times[] = {0.f, 1.f};
  const float values[] = {1.f, 2.f};
  source.addAnimation("archived animation")
      .addObjectParameterBinding(
          geometry.data(), "radius", ANARI_FLOAT32, values, times, 2);

  tsd::core::DataTree tree;
  REQUIRE(tsd::io::serialize_AnimationManagerArchive(source, tree.root()));

  tsd::animation::AnimationManager target(&scene);
  target.addAnimation("stale animation");
  target.play();
  REQUIRE(tsd::io::deserialize_AnimationManagerArchive(target, tree.root()));
  REQUIRE(target.animations().size() == 1);
  REQUIRE(target.animations().front().name() == "archived animation");
  REQUIRE(target.getAnimationTime() == 0.25f);
  REQUIRE(target.getAnimationIncrement() == 0.125f);
  REQUIRE(target.getAnimationTotalFrames() == 41);
  REQUIRE(target.getAnimationFPS() == 24.f);
  REQUIRE_FALSE(target.isLoop());
  REQUIRE_FALSE(target.isPlaying());
  REQUIRE(target.animations().front().objectParameterBindings().front().target()
      == geometry.data());

  const auto filename = (std::filesystem::temp_directory_path()
      / "tsd_animation_manager_archive.tsd")
                            .string();
  REQUIRE(tsd::io::save_AnimationManagerArchive(source, filename.c_str()));
  tsd::animation::AnimationManager fileTarget(&scene);
  REQUIRE(tsd::io::load_AnimationManagerArchive(fileTarget, filename.c_str()));
  REQUIRE(fileTarget.animations().size() == 1);
  std::remove(filename.c_str());

  tsd::core::DataTree invalidTree;
  invalidTree.root() = tree.root();
  invalidTree.root().remove("fps");
  tsd::animation::AnimationManager preserved(&scene);
  preserved.addAnimation("preserved animation");
  REQUIRE_FALSE(tsd::io::deserialize_AnimationManagerArchive(
      preserved, invalidTree.root()));
  REQUIRE(preserved.animations().size() == 1);
  REQUIRE(preserved.animations().front().name() == "preserved animation");

  tsd::scene::Scene incompatibleScene;
  tsd::animation::AnimationManager incompatible(&incompatibleScene);
  incompatible.addAnimation("also preserved");
  REQUIRE_FALSE(
      tsd::io::deserialize_AnimationManagerArchive(incompatible, tree.root()));
  REQUIRE(incompatible.animations().size() == 1);
  REQUIRE(incompatible.animations().front().name() == "also preserved");
}
