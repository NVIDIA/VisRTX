// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Any.hpp"
#include "tsd/core/FlatMap.hpp"
#include "tsd/core/Forest.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/core/TSDTypes.hpp"

namespace tsd::core {

struct Array;
struct Object;
struct Scene;
struct Transform;

struct LayerNodeData
{
  LayerNodeData() = default;
  LayerNodeData(const char *n);
  LayerNodeData(Object *o, const char *n = "");
  LayerNodeData(
      anari::DataType type, size_t index, Scene *s, const char *n = "");
  template <typename T>
  LayerNodeData(ObjectPoolRef<T> obj, const char *n = "");

  LayerNodeData(const LayerNodeData &o);
  LayerNodeData(LayerNodeData &&o);
  LayerNodeData &operator=(const LayerNodeData &o);
  LayerNodeData &operator=(LayerNodeData &&o);

  ~LayerNodeData();

  anari::DataType type() const;
  bool isObject() const;
  bool isTransform() const;
  bool isEmpty() const;
  bool isEnabled() const;

  void setAsObject(anari::DataType type, size_t index, Scene *s);
  void setAsObject(Object *o);
  void setAsTransform(Transform *xfm);
  void setEmpty();

  void setEnabled(bool enabled);

  Object *getObject() const;
  size_t getObjectIndex() const;
  Transform *getTransformObject() const;

  std::string &name();
  const std::string &name() const;

  //////////////////////////////////////////////////////////////////
  // Warning: these operate on the raw Any value, no type checking!
  Any getValueRaw() const;
  void setValueRaw(const Any &v, Scene *scene = nullptr);
  //////////////////////////////////////////////////////////////////

 private:
  void incObjectUseCount();
  void decObjectUseCount();

  // Data //

  std::string m_name;
  bool m_enabled{true};
  Any m_value;
  Scene *m_scene{nullptr};
};

using Layer = Forest<LayerNodeData>;
using LayerVisitor = Layer::Visitor;
using LayerNode = ForestNode<LayerNodeData>;
using LayerNodeRef = LayerNode::Ref;

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline LayerNodeData::LayerNodeData(ObjectPoolRef<T> obj, const char *n)
    : LayerNodeData(obj.data(), n)
{}

} // namespace tsd::core
