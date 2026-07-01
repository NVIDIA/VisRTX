// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/serialization/Layer.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"
// std
#include <algorithm>
#include <stack>
#include <vector>

namespace tsd::io {

namespace {

void serializeLayerNodeInstanceParameters(
    const scene::LayerNodeData &data, core::DataNode &node)
{
  const auto &instanceParameters = data.getInstanceParameters();
  if (instanceParameters.empty())
    return;

  auto &instanceParametersNode = node.append("instanceParameters");
  for (const auto &parameter : instanceParameters)
    instanceParametersNode.append(parameter.first) = parameter.second;
}

void deserializeLayerNodeInstanceParameters(
    core::DataNode &node, scene::LayerNodeData &data)
{
  if (auto *instanceParametersNode = node.child("instanceParameters")) {
    instanceParametersNode->foreach_child([&](core::DataNode &parameterNode) {
      data.setInstanceParameter(parameterNode.name(), parameterNode.getValue());
    });
  }
}

} // namespace

namespace detail {

void serialize_LayerSubtree(const scene::Layer &layer,
    scene::LayerNodeRef start,
    core::DataNode &node,
    const std::vector<scene::LayerNodeRef> *excluded)
{
  const auto isExcludedNode = [&](const scene::LayerNode &layerNode) {
    if (!excluded)
      return false;
    return std::any_of(excluded->begin(),
        excluded->end(),
        [&](scene::LayerNodeRef excludedNode) {
          return excludedNode && &(*excludedNode) == &layerNode;
        });
  };

  std::stack<core::DataNode *> nodes;
  core::DataNode *currentParentNode = nullptr;
  core::DataNode *currentNode = &node;
  int currentLevel = -1;
  layer.traverse_const(
      start, [&](const scene::LayerNode &layerNode, int level) {
        if (isExcludedNode(layerNode))
          return false;

        if (currentLevel < level) {
          nodes.push(currentNode);
          currentParentNode = currentNode;
        } else if (currentLevel > level) {
          for (int i = 0; i < currentLevel - level; i++)
            nodes.pop();
          currentParentNode = nodes.top();
        }

        currentLevel = level;

        if (level == 0)
          currentNode = &node;
        else
          currentNode = &currentParentNode->child("children")->append();

        currentNode->append("name") = layerNode->name();
        currentNode->append("value") = layerNode->getValueRaw();
        if (layerNode->isTransform()) {
          currentNode->append("transformSRT") = layerNode->getTransformSRT();
        }
        currentNode->append("enabled") = layerNode->isEnabled();
        serializeLayerNodeInstanceParameters(*layerNode, *currentNode);
        currentNode->append("children");

        return true;
      });
}

} // namespace detail

void serialize_Layer(const scene::Layer &layer, core::DataNode &node)
{
  serialize_LayerSubtree(layer, layer.root(), node);
}

void serialize_LayerSubtree(
    const scene::Layer &layer, scene::LayerNodeRef start, core::DataNode &node)
{
  detail::serialize_LayerSubtree(layer, start, node, nullptr);
}

void deserialize_Layer(
    core::DataNode &rootNode, scene::Layer &layer, scene::Scene &)
{
  layer.clear();

  std::stack<scene::LayerNodeRef> layerNodes;
  scene::LayerNodeRef currentParentNode;
  scene::LayerNodeRef currentNode = layer.root();
  int currentLevel = -1;
  rootNode.traverse([&](core::DataNode &node, int level) {
    if (level & 0x1 || !node.child("children"))
      return true;

    level /= 2;
    if (currentLevel < level) {
      layerNodes.push(currentNode);
      currentParentNode = currentNode;
    } else if (currentLevel > level) {
      for (int i = 0; i < currentLevel - level; i++)
        layerNodes.pop();
      currentParentNode = layerNodes.top();
    }

    currentLevel = level;

    if (level == 0) {
      currentNode = layer.root();
    } else {
      currentNode = currentParentNode->insert_last_child({&layer});
      if (auto *child = node.child("transformSRT"); child != nullptr)
        (*currentNode)->setAsTransform(child->getValueAs<math::mat3>());
      else
        (*currentNode)->setValueRaw(node["value"].getValue());
      (*currentNode)->setEnabled(node["enabled"].getValueOr(true));
      (*currentNode)->name() = node["name"].getValueAs<std::string>();
      deserializeLayerNodeInstanceParameters(node, (*currentNode).value());
    }

    return true;
  });
}

} // namespace tsd::io
