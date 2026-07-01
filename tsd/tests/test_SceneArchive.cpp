// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/DataTree.hpp"
#include "tsd/io/archives/SceneArchive.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <vector>

SCENARIO(
    "Scene Archives serialize sparse pools observationally", "[SceneArchive]")
{
  tsd::scene::Scene source;
  auto removed = source.createObject<tsd::scene::Geometry>("sphere");
  auto retained = source.createObject<tsd::scene::Geometry>("cylinder");
  retained->setName("retained geometry");
  source.removeObject(removed.data());
  source.insertChildObjectNode(source.defaultLayer()->root(), retained);
  auto objectArray = source.createArray(ANARI_GEOMETRY, 1);
  objectArray->setData(std::vector<size_t>{retained.index()});

  REQUIRE(retained.index() == 1);
  REQUIRE_FALSE(source.objectDB().geometry.is_dense());

  tsd::core::DataTree tree;
  REQUIRE(tsd::io::serialize_SceneArchive(source, tree.root()));

  REQUIRE(retained.index() == 1);
  REQUIRE_FALSE(source.objectDB().geometry.is_dense());
  auto *geometry = tree.root()["objectDB"]["geometry"].child(0);
  REQUIRE(geometry != nullptr);
  REQUIRE((*geometry)["self"].getValue().getAsObjectIndex() == 0);

  tsd::scene::Scene target;
  REQUIRE(tsd::io::deserialize_SceneArchive(target, tree.root()));
  REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 1);
  auto restored = target.getObject<tsd::scene::Geometry>(0);
  REQUIRE(restored);
  REQUIRE(restored->name() == "retained geometry");
  auto restoredArray = target.getObject<tsd::scene::Array>(0);
  REQUIRE(restoredArray);
  REQUIRE(restoredArray->dataAs<size_t>()[0] == 0);

  auto *layer = target.defaultLayer();
  auto child = layer->root()->next();
  REQUIRE(child);
  REQUIRE((*child)->getObject() == restored.data());

  const auto filename =
      (std::filesystem::temp_directory_path() / "tsd_scene_archive.tsd")
          .string();
  REQUIRE(tsd::io::save_SceneArchive(source, filename.c_str()));
  tsd::scene::Scene fileTarget;
  REQUIRE(tsd::io::load_SceneArchive(fileTarget, filename.c_str()));
  REQUIRE(fileTarget.numberOfObjects(ANARI_GEOMETRY) == 1);
  std::remove(filename.c_str());

  tsd::core::DataTree invalidTree;
  invalidTree.root() = tree.root();
  (*invalidTree.root()["objectDB"]["geometry"].child(0))["self"] =
      tsd::core::Any(ANARI_GEOMETRY, size_t(7));
  tsd::scene::Scene preservedTarget;
  preservedTarget.createObject<tsd::scene::Geometry>("cone");
  preservedTarget.addLayer("preserved");
  const auto geometriesBefore = preservedTarget.numberOfObjects(ANARI_GEOMETRY);
  REQUIRE_FALSE(
      tsd::io::deserialize_SceneArchive(preservedTarget, invalidTree.root()));
  REQUIRE(preservedTarget.numberOfObjects(ANARI_GEOMETRY) == geometriesBefore);
  REQUIRE(preservedTarget.layer("preserved") != nullptr);
}

SCENARIO(
    "Scene Archives support full and proxy array carriers", "[SceneArchive]")
{
  tsd::scene::Scene source;
  auto array = source.createArray(ANARI_FLOAT32, 3);
  array->setData(std::vector<float>{1.f, 2.f, 3.f});

  tsd::core::DataTree fullTree;
  REQUIRE(tsd::io::serialize_SceneArchive(
      source, fullTree.root(), tsd::io::ArrayDataPolicy::IncludeData));
  std::vector<std::byte> buffer;
  REQUIRE(fullTree.write(buffer));
  tsd::core::DataTree receivedTree;
  REQUIRE(receivedTree.read(buffer));
  tsd::scene::Scene fullTarget;
  REQUIRE(tsd::io::deserialize_SceneArchive(fullTarget, receivedTree.root()));
  auto fullArray = fullTarget.getObject<tsd::scene::Array>(0);
  REQUIRE(fullArray);
  REQUIRE_FALSE(fullArray->isProxy());
  REQUIRE(fullArray->dataAs<float>()[2] == 3.f);

  tsd::core::DataTree proxyTree;
  REQUIRE(tsd::io::serialize_SceneArchive(
      source, proxyTree.root(), tsd::io::ArrayDataPolicy::ProxyOnly));
  tsd::scene::Scene proxyTarget;
  REQUIRE(tsd::io::deserialize_SceneArchive(proxyTarget, proxyTree.root()));
  auto proxyArray = proxyTarget.getObject<tsd::scene::Array>(0);
  REQUIRE(proxyArray);
  REQUIRE(proxyArray->isProxy());
  REQUIRE(proxyArray->size() == 3);
}
