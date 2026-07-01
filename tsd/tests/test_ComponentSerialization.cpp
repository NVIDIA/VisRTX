// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/DataTree.hpp"
#include "tsd/io/serialization.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <string>
#include <vector>

SCENARIO("Parameter components preserve their DataNode representation",
    "[ComponentSerialization]")
{
  tsd::scene::Scene sourceScene;
  auto sourceObject = sourceScene.createObject<tsd::scene::Geometry>("sphere");
  auto &source = sourceObject->addParameter("attribute");
  source.setValue(0.5f)
      .setDescription("selected attribute")
      .setUsage(tsd::scene::ParameterUsageHint::COLOR)
      .setMin(0.f)
      .setMax(1.f)
      .setStringValues({"attribute0", "attribute1"})
      .setStringSelection(1)
      .setEnabled(false);

  tsd::core::DataTree tree;
  auto &node = tree.root();
  tsd::io::serialize_Parameter(source, node);

  REQUIRE(node.numChildren() == 8);
  REQUIRE(node["value"].getValueAs<float>() == 0.5f);
  REQUIRE_FALSE(node["enabled"].getValueAs<bool>());
  REQUIRE(
      node["description"].getValueAs<std::string>() == "selected attribute");
  REQUIRE(node["usage"].getValueAs<int>()
      == int(tsd::scene::ParameterUsageHint::COLOR));
  REQUIRE(node["min"].getValueAs<float>() == 0.f);
  REQUIRE(node["max"].getValueAs<float>() == 1.f);
  REQUIRE(node["stringValues"].numChildren() == 2);
  REQUIRE(node["stringSelection"].getValueAs<int>() == 1);
  REQUIRE(node.child("__tsd_metadata") == nullptr);

  tsd::scene::Scene targetScene;
  auto targetObject = targetScene.createObject<tsd::scene::Geometry>("sphere");
  auto &target = targetObject->addParameter("attribute");
  tsd::io::deserialize_Parameter(node, target);

  REQUIRE(target.value().getAs<float>() == 0.5f);
  REQUIRE_FALSE(target.isEnabled());
  REQUIRE(target.description() == "selected attribute");
  REQUIRE(target.usage() == tsd::scene::ParameterUsageHint::COLOR);
  REQUIRE(target.min().getAs<float>() == 0.f);
  REQUIRE(target.max().getAs<float>() == 1.f);
  REQUIRE(target.stringValues()
      == std::vector<std::string>{"attribute0", "attribute1"});
  REQUIRE(target.stringSelection() == 1);
}

SCENARIO("Object components preserve their DataNode representation",
    "[ComponentSerialization]")
{
  tsd::scene::Scene sourceScene;
  auto source = sourceScene.createObject<tsd::scene::Geometry>("sphere");
  source->removeAllParameters();
  source->setName("serialized sphere");
  source->setParameter("radius", 2.5f);
  source->setMetadataValue("priority", 3);
  const int bins[] = {1, 2, 3};
  source->setMetadataArray("bins", ANARI_INT32, bins, 3);

  tsd::core::DataTree tree;
  auto &node = tree.root();
  tsd::io::serialize_Object(*source, node);

  REQUIRE(node.numChildren() == 5);
  REQUIRE(node["name"].getValueAs<std::string>() == "serialized sphere");
  REQUIRE(node["self"].getValue().type() == ANARI_GEOMETRY);
  REQUIRE(node["self"].getValue().getAsObjectIndex() == source->index());
  REQUIRE(node["subtype"].getValueAs<std::string>() == "sphere");
  REQUIRE(node["parameters"].numChildren() == 1);
  REQUIRE(node["parameters"]["radius"]["value"].getValueAs<float>() == 2.5f);
  REQUIRE(node["metadata"].numChildren() == 2);
  REQUIRE(node.child("objectDB") == nullptr);
  REQUIRE(node.child("__tsd_metadata") == nullptr);

  tsd::scene::Scene targetScene;
  tsd::io::deserialize_Object(targetScene, node);
  auto target = targetScene.getObject<tsd::scene::Geometry>(source->index());

  REQUIRE(target);
  REQUIRE(target->name() == "serialized sphere");
  REQUIRE(target->parameterValueAs<float>("radius") == 2.5f);
  REQUIRE(target->getMetadataValue("priority").getAs<int>() == 3);
  anari::DataType binsType = ANARI_UNKNOWN;
  const void *binsData = nullptr;
  size_t numBins = 0;
  target->getMetadataArray("bins", &binsType, &binsData, &numBins);
  REQUIRE(binsType == ANARI_INT32);
  REQUIRE(numBins == 3);
  const auto *targetBins = static_cast<const int *>(binsData);
  REQUIRE(targetBins[0] == 1);
  REQUIRE(targetBins[2] == 3);
}

SCENARIO("Layer components preserve their DataNode representation",
    "[ComponentSerialization]")
{
  tsd::scene::Scene sourceScene;
  auto sourceObject = sourceScene.createObject<tsd::scene::Geometry>("sphere");
  auto *sourceLayer = sourceScene.defaultLayer();
  auto group = sourceScene.insertChildTransformNode(
      sourceLayer->root(), tsd::math::IDENTITY_MAT4, "group");
  (*group)->setEnabled(false);
  auto instance =
      sourceScene.insertChildObjectNode(group, sourceObject, "sphere instance");
  (*instance).value().setInstanceParameter("opacity", 0.25f);

  tsd::core::DataTree tree;
  auto &node = tree.root();
  tsd::io::serialize_Layer(*sourceLayer, node);

  REQUIRE(node.numChildren() == 5);
  REQUIRE(node.child("name") != nullptr);
  REQUIRE(node.child("value") != nullptr);
  REQUIRE(node.child("enabled") != nullptr);
  auto *children = node.child("children");
  REQUIRE(children != nullptr);
  REQUIRE(children->numChildren() == 1);
  auto *groupNode = children->child(0);
  REQUIRE(groupNode != nullptr);
  REQUIRE(groupNode->numChildren() == 5);
  REQUIRE(groupNode->child("name")->getValueAs<std::string>() == "group");
  REQUIRE(groupNode->child("transformSRT") != nullptr);
  REQUIRE_FALSE(groupNode->child("enabled")->getValueAs<bool>());
  auto *instanceNode = groupNode->child("children")->child(0);
  REQUIRE(instanceNode != nullptr);
  REQUIRE(instanceNode->numChildren() == 5);
  REQUIRE(instanceNode->child("name")->getValueAs<std::string>()
      == "sphere instance");
  REQUIRE(instanceNode->child("value")->getValue().type() == ANARI_GEOMETRY);
  REQUIRE(instanceNode->child("instanceParameters")->numChildren() == 1);
  REQUIRE(instanceNode->child("instanceParameters")
              ->child("opacity")
              ->getValueAs<float>()
      == 0.25f);
  REQUIRE(instanceNode->child("children")->numChildren() == 0);
  REQUIRE(node.child("objectDB") == nullptr);
  REQUIRE(node.child("__tsd_metadata") == nullptr);

  tsd::scene::Scene targetScene;
  targetScene.createObject<tsd::scene::Geometry>("sphere");
  auto *targetLayer = targetScene.defaultLayer();
  tsd::io::deserialize_Layer(node, *targetLayer, targetScene);

  bool sawGroup = false;
  bool sawInstance = false;
  targetLayer->traverse(
      targetLayer->root(), [&](tsd::scene::LayerNode &layerNode, int level) {
        if (level == 1) {
          sawGroup = true;
          REQUIRE(layerNode.value().name() == "group");
          REQUIRE_FALSE(layerNode.value().isEnabled());
        } else if (level == 2) {
          sawInstance = true;
          REQUIRE(layerNode.value().name() == "sphere instance");
          REQUIRE(layerNode.value()
                      .getInstanceParameters()
                      .at("opacity")
                      ->getAs<float>()
              == 0.25f);
        }
        return true;
      });
  REQUIRE(sawGroup);
  REQUIRE(sawInstance);
}
