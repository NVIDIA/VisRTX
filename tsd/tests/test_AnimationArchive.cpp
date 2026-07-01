// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/io/archives/AnimationArchive.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <cstdio>
#include <filesystem>

SCENARIO("Animation Archives add only when Scene bindings are compatible",
    "[AnimationArchive]")
{
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager source(&scene);
  auto geometry = scene.createObject<tsd::scene::Geometry>("sphere");
  const float times[] = {0.f, 1.f};
  const float values[] = {1.f, 2.f};
  auto &animation = source.addAnimation("radius");
  animation.addObjectParameterBinding(
      geometry.data(), "radius", ANARI_FLOAT32, values, times, 2);

  tsd::core::DataTree tree;
  REQUIRE(tsd::io::serialize_AnimationArchive(animation, tree.root()));

  tsd::animation::AnimationManager target(&scene);
  auto *restored = tsd::io::deserialize_AnimationArchive(target, tree.root());
  REQUIRE(restored != nullptr);
  REQUIRE(restored->name() == "radius");
  REQUIRE(restored->objectParameterBindings().size() == 1);
  REQUIRE(
      restored->objectParameterBindings().front().target() == geometry.data());

  tsd::core::DataTree invalidTree;
  invalidTree.root() = tree.root();
  (*invalidTree.root()["objectBindings"].child(0))["targetIndex"] = size_t(99);
  const auto countBefore = target.animations().size();
  REQUIRE(tsd::io::deserialize_AnimationArchive(target, invalidTree.root())
      == nullptr);
  REQUIRE(target.animations().size() == countBefore);

  tsd::core::DataTree malformedTree;
  malformedTree.root() = tree.root();
  (*malformedTree.root()["objectBindings"].child(0))["targetIndex"] =
      "not an index";
  REQUIRE(tsd::io::deserialize_AnimationArchive(target, malformedTree.root())
      == nullptr);
  REQUIRE(target.animations().size() == countBefore);

  const auto filename =
      (std::filesystem::temp_directory_path() / "tsd_animation_archive.tsd")
          .string();
  REQUIRE(tsd::io::save_AnimationArchive(animation, filename.c_str()));
  tsd::animation::AnimationManager fileTarget(&scene);
  REQUIRE(tsd::io::load_AnimationArchive(fileTarget, filename.c_str()));
  std::remove(filename.c_str());
}
