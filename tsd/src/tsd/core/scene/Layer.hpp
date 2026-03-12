// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/LayerNodeData.hpp"

namespace tsd::core {

struct Scene;

struct Layer final
{
  using Visitor = ForestVisitor<LayerNodeData>;
  using VisitorEntryFunction = ForestVisitorEntryFunction<LayerNodeData>;
  using VisitorExitFunction = ForestVisitorExitFunction<LayerNodeData>;

  Layer(Scene *scene, std::string name = "<unnamed layer>");
  ~Layer();

  Scene *scene() const;

  const std::string &name() const;
  std::string &editableName();

  size_t size() const;
  size_t capacity() const;
  bool empty() const;

  LayerNodeRef root() const;
  LayerNodeRef at(size_t i) const;
  void erase(LayerNodeRef obj);
  void clear();

  bool isAncestorOf(LayerNodeRef potentialAncestor, LayerNodeRef node) const;

  void traverse(LayerNodeRef start, Visitor &visitor);
  void traverse(LayerNodeRef start, VisitorEntryFunction &&f);
  void traverse(LayerNodeRef start,
      VisitorEntryFunction &&onNodeEntry,
      VisitorExitFunction &&onNodeExit);

 private:
  Scene *m_scene{nullptr};
  std::string m_name;
  Forest<LayerNodeData> m_tree{
      {this, tsd::math::mat4(tsd::math::identity), "root"}};
};

using LayerVisitor = Layer::Visitor;

} // namespace tsd::core
