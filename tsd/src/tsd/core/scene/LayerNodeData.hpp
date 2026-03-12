// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Any.hpp"
#include "tsd/core/FlatMap.hpp"
#include "tsd/core/Forest.hpp"
#include "tsd/core/TSDMath.hpp"

namespace tsd::core {

struct Array;
struct Object;
struct Layer;
using InstanceParameterMap = FlatMap<std::string, Any>;

struct LayerNodeData
{
  LayerNodeData() = default;
  LayerNodeData(Layer *layer, const char *n = "");
  LayerNodeData(Layer *layer, const math::mat4 &m, const char *n = "");
  LayerNodeData(Layer *layer, const math::mat3 &m, const char *n = "");
  LayerNodeData(Layer *layer, Object *o, const char *n = "");
  template <typename T>
  LayerNodeData(Layer *layer, ObjectPoolRef<T> obj, const char *n = "");

  LayerNodeData(const LayerNodeData &o);
  LayerNodeData(LayerNodeData &&o);
  LayerNodeData &operator=(const LayerNodeData &o);
  LayerNodeData &operator=(LayerNodeData &&o);

  ~LayerNodeData();

  const Layer *layer() const;
  Layer *layer();

  bool hasDefault() const;
  bool isDefaultValue() const;
  void setToDefaultValue();
  void setCurrentValueAsDefault();

  anari::DataType type() const;
  bool isObject() const;
  bool isTransform() const;
  bool isEmpty() const;
  bool isEnabled() const;

  void setAsObject(Object *o);
  void setAsObject(anari::DataType type, size_t index);
  void setAsTransform(const math::mat4 &m);
  void setAsTransform(const math::mat4 &m, const math::mat4 &defaultM);
  void setAsTransform(const math::mat3 &srt);
  void setAsTransformArray(Array *a);
  void setEmpty();

  void setEnabled(bool enabled);

  Object *getObject() const;
  size_t getObjectIndex() const;
  math::mat4 getTransform() const;
  math::mat3 getTransformSRT() const;
  Array *getTransformArray() const;

  std::string &name();
  const std::string &name() const;

  //////////////////////////////////////////////////////////////////
  // Warning: these operate on the raw Any value, no type checking!
  Any getValueRaw() const;
  void setValueRaw(const Any &v);
  //////////////////////////////////////////////////////////////////

  const InstanceParameterMap &getInstanceParameters() const;
  void setInstanceParameter(const std::string &name, Any v);
  void clearInstanceParameters();

 private:
  void incObjectUseCount();
  void decObjectUseCount();

  // Data //

  std::string m_name;
  bool m_enabled{true};
  Any m_value;
  Any m_defaultValue;
  math::mat3 m_srt; // scale, azelrot, translation
  InstanceParameterMap m_instanceParameters;
  Layer *m_layer{nullptr};
};

using LayerNode = ForestNode<LayerNodeData>;
using LayerNodeRef = LayerNode::Ref;

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline LayerNodeData::LayerNodeData(
    Layer *layer, ObjectPoolRef<T> obj, const char *n)
    : LayerNodeData(layer, obj.data(), n)
{}

} // namespace tsd::core
