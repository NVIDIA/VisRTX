// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/FlatMap.hpp"
#include "tsd/core/Forest.hpp"
#include "tsd/core/Any.hpp"
#include "tsd/core/TSDMath.hpp"

namespace tsd::core {

struct Context;
struct Object;

struct LayerNodeData
{
  LayerNodeData() = default;
  template <typename T>
  LayerNodeData(T v, const char *n = "");
  LayerNodeData(Any v, const char *n);

  bool hasDefault() const;
  bool isObject() const;
  bool isTransform() const;
  bool isEmpty() const;

  Object *getObject(Context *ctx) const;
  math::mat4 getTransform() const;

  // Data //

  Any value;
  Any defaultValue;
  std::string name;
  bool enabled{true};
  FlatMap<std::string, Any> customParameters;
  FlatMap<std::string, Any> valueCache;
};

using Layer = Forest<LayerNodeData>;
using LayerVisitor = Layer::Visitor;
using LayerNode = ForestNode<LayerNodeData>;
using LayerNodeRef = LayerNode::Ref;

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline LayerNodeData::LayerNodeData(T v, const char *n)
    : LayerNodeData(Any(v), n)
{}

} // namespace tsd::core
