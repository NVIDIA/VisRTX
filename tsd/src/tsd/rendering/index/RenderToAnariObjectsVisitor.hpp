// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/AnariObjectCache.hpp"
#include "tsd/core/TSDTypes.hpp"
#include "tsd/core/scene/Layer.hpp"
#include "tsd/core/scene/objects/Transform.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndexFilterFcn.hpp"
// std
#include <anari/anari_cpp.hpp>
#include <cassert>
#include <stack>
#include <vector>

namespace tsd::rendering {

struct RenderToAnariObjectsVisitor : public tsd::core::LayerVisitor
{
  RenderToAnariObjectsVisitor(anari::Device d,
      tsd::core::AnariObjectCache &cache,
      std::vector<anari::Instance> &instanceCache,
      anari::Instance &rootInstance,
      RenderIndexFilterFcn filter);
  ~RenderToAnariObjectsVisitor();

  bool preChildren(tsd::core::LayerNode &n, int /*level*/) override;
  void postChildren(tsd::core::LayerNode &n, int /*level*/) override;

  void finalizeRootObjects();

 private:
  bool isIncludedAfterFiltering(const tsd::core::LayerNode &n) const;
  bool commitGroupToInstance(anari::Instance instance);

  struct GroupedObjects
  {
    std::vector<anari::Surface> surfaces;
    std::vector<anari::Volume> volumes;
    std::vector<anari::Light> lights;
  };

  anari::Device m_device{nullptr};
  tsd::core::AnariObjectCache *m_cache{nullptr};
  std::vector<anari::Instance> *m_instances;
  anari::Instance *m_rootInstance{nullptr};
  RenderIndexFilterFcn m_filter;
  std::stack<GroupedObjects> m_objects;
  size_t m_nextTransformOrderedIndex{0};
  std::stack<size_t> m_transformOrderedIndices;
  size_t m_expectedTransformCount{0};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline RenderToAnariObjectsVisitor::RenderToAnariObjectsVisitor(anari::Device d,
    tsd::core::AnariObjectCache &cache,
    std::vector<anari::Instance> &instanceCache,
    anari::Instance &rootInstance,
    RenderIndexFilterFcn filter)
    : m_device(d),
      m_cache(&cache),
      m_instances(&instanceCache),
      m_rootInstance(&rootInstance),
      m_filter(std::move(filter)),
      m_expectedTransformCount(instanceCache.size())
{
  anari::retain(d, d);
  m_objects.emplace();
}

inline RenderToAnariObjectsVisitor::~RenderToAnariObjectsVisitor()
{
  anari::release(m_device, m_device);
}

inline bool RenderToAnariObjectsVisitor::preChildren(
    tsd::core::LayerNode &n, int /*level*/)
{
  if (!n->isEnabled())
    return false;

  auto &current = m_objects.top();

  const bool included = isIncludedAfterFiltering(n);

  auto type = n->type();

  switch (type) {
  case ANARI_SURFACE: {
    size_t i = n->getObjectIndex();
    if (auto h = m_cache->getHandle(type, i, true); h != nullptr && included)
      current.surfaces.push_back((anari::Surface)h);
    break;
  }
  case ANARI_VOLUME: {
    size_t i = n->getObjectIndex();
    if (auto h = m_cache->getHandle(type, i, true); h != nullptr && included)
      current.volumes.push_back((anari::Volume)h);
    break;
  }
  case ANARI_LIGHT: {
    size_t i = n->getObjectIndex();
    if (auto h = m_cache->getHandle(type, i, true); h != nullptr)
      current.lights.push_back((anari::Light)h);
    break;
  }
  case core::TSD_TRANSFORM: {
    assert(m_nextTransformOrderedIndex < m_instances->size());
    m_transformOrderedIndices.push(m_nextTransformOrderedIndex++);
    m_objects.emplace();
    break;
  }
  }

  return true;
}

inline void RenderToAnariObjectsVisitor::postChildren(
    tsd::core::LayerNode &n, int /*level*/)
{
  if (!n->isEnabled())
    return;

  auto nodeType = n->type();
  switch (nodeType) {
  case core::TSD_TRANSFORM: {
    const auto orderedIndex = m_transformOrderedIndices.top();
    m_transformOrderedIndices.pop();
    assert(orderedIndex < m_instances->size());

    if (!commitGroupToInstance((*m_instances)[orderedIndex])) {
      anari::release(m_device, (*m_instances)[orderedIndex]);
      (*m_instances)[orderedIndex] = {};
    }

    m_objects.pop();
    break;
  }
  }
}

inline bool RenderToAnariObjectsVisitor::isIncludedAfterFiltering(
    const tsd::core::LayerNode &n) const
{
  if (!m_filter)
    return true;

  auto type = n->type();
  if (!anari::isObject(type) && !tsd::core::isTSDTransform(type))
    return false;

  return m_filter(n->getObject());
}

inline bool RenderToAnariObjectsVisitor::commitGroupToInstance(
    anari::Instance instance)
{
  auto &current = m_objects.top();

  if (current.surfaces.empty() && current.volumes.empty()
      && current.lights.empty()) {
    anari::unsetParameter(m_device, instance, "group");
    anari::commitParameters(m_device, instance);
    return false;
  }

  auto group = anari::newObject<anari::Group>(m_device);

  if (!current.surfaces.empty()) {
    anari::setParameterArray1D(m_device,
        group,
        "surface",
        current.surfaces.data(),
        current.surfaces.size());
  }

  if (!current.volumes.empty()) {
    anari::setParameterArray1D(m_device,
        group,
        "volume",
        current.volumes.data(),
        current.volumes.size());
  }

  if (!current.lights.empty()) {
    anari::setParameterArray1D(
        m_device, group, "light", current.lights.data(), current.lights.size());
  }

  anari::commitParameters(m_device, group);

  anari::setParameter(m_device, instance, "group", group);
  anari::commitParameters(m_device, instance);

  anari::release(m_device, group);

  return true;
}

inline void RenderToAnariObjectsVisitor::finalizeRootObjects()
{
  assert(m_transformOrderedIndices.empty());
  assert(m_nextTransformOrderedIndex == m_expectedTransformCount);
  if (!commitGroupToInstance(*m_rootInstance)) {
    anari::release(m_device, *m_rootInstance);
    *m_rootInstance = {};
  }
}

} // namespace tsd::rendering
