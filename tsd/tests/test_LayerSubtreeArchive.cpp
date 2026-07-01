// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/DataTree.hpp"
#include "tsd/io/archives/LayerSubtreeArchive.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <cstdio>
#include <filesystem>

SCENARIO("Layer Subtree Archives require a destination and add atomically",
    "[LayerSubtreeArchive]")
{
  tsd::scene::Scene source;
  auto geometry = source.createObject<tsd::scene::Geometry>("sphere");
  auto material = source.createObject<tsd::scene::Material>("matte");
  auto surface = source.createSurface("surface", geometry, material);
  auto subtree =
      source.insertChildNode(source.defaultLayer()->root(), "subtree");
  source.insertChildObjectNode(subtree, surface, "instance");

  tsd::core::DataTree tree;
  REQUIRE(tsd::io::serialize_LayerSubtreeArchive(subtree, tree.root()));
  REQUIRE(tree.root().child("animations") == nullptr);

  tsd::scene::Scene target;
  const auto surfacesBefore = target.numberOfObjects(ANARI_SURFACE);
  REQUIRE_FALSE(tsd::io::deserialize_LayerSubtreeArchive({}, tree.root()));
  REQUIRE(target.numberOfObjects(ANARI_SURFACE) == surfacesBefore);

  auto restored = tsd::io::deserialize_LayerSubtreeArchive(
      target.defaultLayer()->root(), tree.root());
  REQUIRE(restored);
  REQUIRE((*restored)->name() == "subtree");
  REQUIRE(target.numberOfObjects(ANARI_SURFACE) == surfacesBefore + 1);

  tsd::core::DataTree invalidTree;
  invalidTree.root() = tree.root();
  invalidTree.root()["subtree"]["value"] =
      tsd::core::Any(ANARI_SURFACE, size_t(99));
  const auto objectsBeforeFailure = target.numberOfObjects(ANARI_SURFACE);
  REQUIRE_FALSE(tsd::io::deserialize_LayerSubtreeArchive(
      target.defaultLayer()->root(), invalidTree.root()));
  REQUIRE(target.numberOfObjects(ANARI_SURFACE) == objectsBeforeFailure);

  const auto filename =
      (std::filesystem::temp_directory_path() / "tsd_layer_subtree_archive.tsd")
          .string();
  REQUIRE(tsd::io::save_LayerSubtreeArchive(subtree, filename.c_str()));
  tsd::scene::Scene fileTarget;
  REQUIRE(tsd::io::load_LayerSubtreeArchive(
      fileTarget.defaultLayer()->root(), filename.c_str()));
  std::remove(filename.c_str());

  tsd::core::DataTree legacyTree;
  legacyTree.root() = tree.root();
  legacyTree.root().remove("__tsd_metadata");
  tsd::scene::Scene legacyTarget;
  REQUIRE(tsd::io::deserialize_LayerSubtreeArchive(
      legacyTarget.defaultLayer()->root(), legacyTree.root()));
}
