// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/Layer.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/scene/Scene.hpp"

namespace tsd::core {

LayerNodeData::LayerNodeData(const char *n)
{
  setEmpty();
  m_name = n;
}

LayerNodeData::LayerNodeData(Object *o, const char *n) : LayerNodeData(n)
{
  setAsObject(o);
}

LayerNodeData::LayerNodeData(
    anari::DataType type, size_t index, Scene *s, const char *n)
    : LayerNodeData(n)
{
  setAsObject(type, index, s);
}

LayerNodeData::LayerNodeData(const LayerNodeData &o)
{
  m_name = o.m_name;
  m_enabled = o.m_enabled;
  m_value = o.m_value;
  m_scene = o.m_scene;
  incObjectUseCount();
}

LayerNodeData::LayerNodeData(LayerNodeData &&o)
{
  m_name = std::move(o.m_name);
  m_enabled = std::move(o.m_enabled);
  m_value = std::move(o.m_value);
  m_scene = std::move(o.m_scene);
  o.m_scene = nullptr;
  o.m_value.reset();
}

LayerNodeData &LayerNodeData::operator=(const LayerNodeData &o)
{
  decObjectUseCount();
  m_name = o.m_name;
  m_enabled = o.m_enabled;
  m_value = o.m_value;
  m_scene = o.m_scene;
  incObjectUseCount();
  return *this;
}

LayerNodeData &LayerNodeData::operator=(LayerNodeData &&o)
{
  decObjectUseCount();
  m_name = std::move(o.m_name);
  m_enabled = std::move(o.m_enabled);
  m_value = std::move(o.m_value);
  m_scene = std::move(o.m_scene);
  o.m_scene = nullptr;
  o.m_value.reset();
  return *this;
}

LayerNodeData::~LayerNodeData()
{
  decObjectUseCount();
}

anari::DataType LayerNodeData::type() const
{
  return m_value.type();
}

bool LayerNodeData::isObject() const
{
  return anari::isObject(type()) || isTSDTransform(type());
}

bool LayerNodeData::isTransform() const
{
  return type() == TSD_TRANSFORM;
}

bool LayerNodeData::isEmpty() const
{
  return !m_value;
}

bool LayerNodeData::isEnabled() const
{
  return m_enabled;
}

void LayerNodeData::setAsObject(Object *o)
{
  if (o)
    setAsObject(o->type(), o->index(), o->scene());
  else {
    tsd::core::logWarning(
        "LayerNodeData::setAsObject() called with null object,"
        " setting to empty");
    setEmpty();
  }
}

void LayerNodeData::setAsObject(anari::DataType type, size_t index, Scene *s)
{
  decObjectUseCount();
  m_value = Any(type, index);
  m_scene = s;
  incObjectUseCount();
}

void LayerNodeData::setEmpty()
{
  decObjectUseCount();
  m_value.reset();
  m_scene = nullptr;
  m_name.clear();
}

void LayerNodeData::setEnabled(bool e)
{
  m_enabled = e;
}

Object *LayerNodeData::getObject() const
{
  return isObject() && m_scene ? m_scene->getObject(m_value) : nullptr;
}

size_t LayerNodeData::getObjectIndex() const
{
  return m_value.getAsObjectIndex();
}

std::string &LayerNodeData::name()
{
  return m_name;
}

const std::string &LayerNodeData::name() const
{
  return m_name;
}

Any LayerNodeData::getValueRaw() const
{
  return m_value;
}

void LayerNodeData::setValueRaw(const Any &v, Scene *scene)
{
  setEmpty();
  m_scene = scene;
  m_value = v;
  incObjectUseCount();
}

Transform *LayerNodeData::getTransformObject() const
{
  if (type() != TSD_TRANSFORM || !m_scene)
    return nullptr;
  return static_cast<Transform *>(
      m_scene->getObject(type(), m_value.getAsObjectIndex()));
}

void LayerNodeData::incObjectUseCount()
{
  if (auto *o = getObject(); o)
    o->incUseCount(Object::UseKind::LAYER);
}

void LayerNodeData::decObjectUseCount()
{
  if (auto *o = getObject(); o)
    o->decUseCount(Object::UseKind::LAYER);
}

} // namespace tsd::core
