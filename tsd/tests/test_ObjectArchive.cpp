// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/DataTree.hpp"
#include "tsd/io/archives/ObjectArchive.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <cstdio>
#include <filesystem>

SCENARIO(
    "Object Archives add a dependency closure atomically", "[ObjectArchive]")
{
  tsd::scene::Scene source;
  auto geometry = source.createObject<tsd::scene::Geometry>("sphere");
  geometry->setName("archived geometry");
  auto material = source.createObject<tsd::scene::Material>("matte");
  auto surface = source.createSurface("archived surface", geometry, material);

  tsd::core::DataTree tree;
  REQUIRE(tsd::io::serialize_ObjectArchive(*surface, tree.root()));

  tsd::scene::Scene target;
  const auto geometryCount = target.numberOfObjects(ANARI_GEOMETRY);
  auto *restored = tsd::io::deserialize_ObjectArchive(target, tree.root());
  REQUIRE(restored != nullptr);
  REQUIRE(restored->type() == ANARI_SURFACE);
  REQUIRE(restored->name() == "archived surface");
  REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == geometryCount + 1);

  tsd::core::DataTree invalidTree;
  invalidTree.root() = tree.root();
  auto &invalidRoot = invalidTree.root();
  auto *geometryNode = invalidRoot["objectDB"]["geometry"].child(0);
  REQUIRE(geometryNode != nullptr);
  (*geometryNode)["parameters"]["missing"]["value"] =
      tsd::core::Any(ANARI_ARRAY1D, size_t(99));

  const auto arraysBefore = target.numberOfObjects(ANARI_ARRAY);
  const auto geometriesBefore = target.numberOfObjects(ANARI_GEOMETRY);
  const auto materialsBefore = target.numberOfObjects(ANARI_MATERIAL);
  const auto surfacesBefore = target.numberOfObjects(ANARI_SURFACE);
  REQUIRE(tsd::io::deserialize_ObjectArchive(target, invalidRoot) == nullptr);
  REQUIRE(target.numberOfObjects(ANARI_ARRAY) == arraysBefore);
  REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == geometriesBefore);
  REQUIRE(target.numberOfObjects(ANARI_MATERIAL) == materialsBefore);
  REQUIRE(target.numberOfObjects(ANARI_SURFACE) == surfacesBefore);

  const auto filename =
      (std::filesystem::temp_directory_path() / "tsd_object_archive.tsd")
          .string();
  REQUIRE(tsd::io::save_ObjectArchive(*surface, filename.c_str()));
  tsd::scene::Scene fileTarget;
  REQUIRE(tsd::io::load_ObjectArchive(fileTarget, filename.c_str()));
  std::remove(filename.c_str());

  tsd::core::DataTree legacyTree;
  legacyTree.root() = tree.root();
  legacyTree.root().remove("__tsd_metadata");
  tsd::scene::Scene legacyTarget;
  REQUIRE(tsd::io::deserialize_ObjectArchive(legacyTarget, legacyTree.root()));
}
