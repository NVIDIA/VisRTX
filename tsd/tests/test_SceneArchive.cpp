// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/io/archives/AnimationManagerArchive.hpp"
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

SCENARIO("Scene and Animation Manager Archives share dense mappings",
    "[SceneArchive]")
{
  tsd::scene::Scene source;
  tsd::animation::AnimationManager sourceAnimations(&source);

  auto removedGeometry = source.createObject<tsd::scene::Geometry>("sphere");
  auto retainedGeometry = source.createObject<tsd::scene::Geometry>("cylinder");
  retainedGeometry->setName("retained geometry");
  source.removeObject(removedGeometry.data());
  REQUIRE(retainedGeometry.index() == 1);

  auto removedMaterial = source.createObject<tsd::scene::Material>("matte");
  auto retainedMaterial = source.createObject<tsd::scene::Material>("matte");
  retainedMaterial->setName("retained material");
  source.removeObject(removedMaterial.data());
  REQUIRE(retainedMaterial.index() == 2);

  auto removedTransform = source.insertChildTransformNode(
      source.defaultLayer()->root(), tsd::math::IDENTITY_MAT4, "removed");
  auto retainedTransform = source.insertChildTransformNode(
      source.defaultLayer()->root(), tsd::math::IDENTITY_MAT4, "retained");
  source.removeNode(removedTransform);
  REQUIRE(retainedTransform.index() == 2);

  const float times[] = {0.f, 1.f};
  tsd::scene::Object *keyframes[] = {
      retainedMaterial.data(), retainedMaterial.data()};
  auto &animation = sourceAnimations.addAnimation("sparse animation");
  animation.addObjectParameterBinding(
      retainedGeometry.data(), "material", ANARI_MATERIAL, keyframes, times, 2);
  animation.addTransformBinding(retainedTransform);

  tsd::core::DataTree sceneTree;
  tsd::core::DataTree animationTree;
  REQUIRE(tsd::io::serialize_SceneAndAnimationManagerArchives(
      source, sourceAnimations, sceneTree.root(), animationTree.root()));

  auto *serializedAnimation = animationTree.root()["objects"].child(0);
  REQUIRE(serializedAnimation != nullptr);
  auto *objectBinding = (*serializedAnimation)["objectBindings"].child(0);
  REQUIRE(objectBinding != nullptr);
  REQUIRE((*objectBinding)["targetIndex"].getValueAs<size_t>() == 0);
  const void *serializedKeyframes = nullptr;
  size_t numSerializedKeyframes = 0;
  anari::DataType serializedKeyframeType = ANARI_UNKNOWN;
  (*objectBinding)["data"].getValueAsArray(
      &serializedKeyframeType, &serializedKeyframes, &numSerializedKeyframes);
  REQUIRE(serializedKeyframeType == ANARI_MATERIAL);
  REQUIRE(numSerializedKeyframes == 2);
  REQUIRE(static_cast<const size_t *>(serializedKeyframes)[0] == 1);

  auto *transformBinding = (*serializedAnimation)["transformBindings"].child(0);
  REQUIRE(transformBinding != nullptr);
  REQUIRE((*transformBinding)["nodeIndex"].getValueAs<size_t>() == 1);

  tsd::scene::Scene target;
  tsd::animation::AnimationManager targetAnimations(&target);
  REQUIRE(tsd::io::deserialize_SceneArchive(target, sceneTree.root()));
  REQUIRE(tsd::io::deserialize_AnimationManagerArchive(
      targetAnimations, animationTree.root()));

  auto restoredGeometry = target.getObject<tsd::scene::Geometry>(0);
  auto restoredMaterial = target.getObject<tsd::scene::Material>(1);
  REQUIRE(restoredGeometry);
  REQUIRE(restoredMaterial);
  REQUIRE(restoredGeometry->name() == "retained geometry");
  REQUIRE(restoredMaterial->name() == "retained material");
  REQUIRE(targetAnimations.animations().size() == 1);
  const auto &restoredAnimation = targetAnimations.animations().front();
  REQUIRE(restoredAnimation.objectParameterBindings().front().target()
      == restoredGeometry.data());
  const auto *restoredKeyframes = static_cast<const size_t *>(
      restoredAnimation.objectParameterBindings().front().data().data());
  REQUIRE(restoredKeyframes[0] == restoredMaterial.index());
  REQUIRE((*restoredAnimation.transformBindings().front().target())->name()
      == "retained");
}
