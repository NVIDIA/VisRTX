// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/containers/FlatMap.hpp"
#include "tsd/containers/Forest.hpp"
#include "tsd/core/Any.hpp"
#include "tsd/core/TSDMath.hpp"

namespace tsd {

struct Context;
struct Object;

struct LayerNodeData
{
  LayerNodeData() = default;
  template <typename T>
  LayerNodeData(T v, const char *n = "");
  LayerNodeData(utility::Any v, const char *n);

  bool hasDefault() const;
  bool isObject() const;
  bool isTransform() const;
  bool isEmpty() const;

  Object *getObject(Context *ctx) const;
  math::mat4 getTransform() const;

  // Data //

  utility::Any value;
  utility::Any defaultValue;
  std::string name;
  bool enabled{true};
  FlatMap<std::string, utility::Any> customParameters;
  FlatMap<std::string, utility::Any> valueCache;
};

using Layer = utility::Forest<LayerNodeData>;
using LayerVisitor = Layer::Visitor;
using LayerNode = utility::ForestNode<LayerNodeData>;
using LayerNodeRef = LayerNode::Ref;

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline LayerNodeData::LayerNodeData(T v, const char *n)
    : LayerNodeData(utility::Any(v), n)
{}

} // namespace tsd
