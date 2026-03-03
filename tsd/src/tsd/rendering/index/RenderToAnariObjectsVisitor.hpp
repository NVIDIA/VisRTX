// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/scene/Layer.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndexFilterFcn.hpp"
// std
#include <algorithm>
#include <anari/anari_cpp.hpp>
#include <cstdint>
#include <iterator>
#include <stack>

namespace tsd::rendering {

enum RenderInclusionMask
{
  SURFACES = 1 << 0,
  VOLUMES = 1 << 1,
  LIGHTS = 1 << 2,
  ALL = ((1 << 3) - 1)
};

constexpr uint8_t objectMask_none()
{
  return 0;
}

constexpr uint8_t objectMask_all()
{
  return RenderInclusionMask::ALL;
}

constexpr uint8_t objectMask_surfacesAndVolumes()
{
  return RenderInclusionMask::SURFACES | RenderInclusionMask::VOLUMES;
}

constexpr uint8_t objectMask_lights()
{
  return RenderInclusionMask::LIGHTS;
}

///////////////////////////////////////////////////////////////////////////////

struct RenderToAnariObjectsVisitor : public tsd::core::LayerVisitor
{
  RenderToAnariObjectsVisitor(anari::Device d,
      tsd::core::AnariObjectCache &oc,
      std::vector<anari::Instance> *instances,
      uint8_t inclusionMask = objectMask_all(),
      RenderIndexFilterFcn *f = nullptr);
  ~RenderToAnariObjectsVisitor();

  bool preChildren(tsd::core::LayerNode &n, int level) override;
  void postChildren(tsd::core::LayerNode &n, int level) override;

 private:
  bool isIncludedAfterFiltering(const tsd::core::LayerNode &n) const;
  anari::Instance createInstanceFromTop();

  struct GroupedObjects
  {
    std::vector<anari::Surface> surfaces;
    std::vector<anari::Volume> volumes;
    std::vector<anari::Light> lights;
  };

  anari::Device m_device{nullptr};
  tsd::core::AnariObjectCache *m_cache{nullptr};
  std::vector<anari::Instance> *m_instances{nullptr};
  std::stack<GroupedObjects> m_objects;
  uint8_t m_mask{objectMask_none()};
  RenderIndexFilterFcn *m_filter{nullptr};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline RenderToAnariObjectsVisitor::RenderToAnariObjectsVisitor(anari::Device d,
    tsd::core::AnariObjectCache &oc,
    std::vector<anari::Instance> *instances,
    uint8_t mask,
    RenderIndexFilterFcn *f)
    : m_device(d),
      m_cache(&oc),
      m_instances(instances),
      m_mask(mask),
      m_filter(f)
{
  anari::retain(d, d);
  m_objects.emplace();
}

inline RenderToAnariObjectsVisitor::~RenderToAnariObjectsVisitor()
{
  anari::release(m_device, m_device);
}

inline bool RenderToAnariObjectsVisitor::preChildren(
    tsd::core::LayerNode &n, int level)
{
  if (!n->isEnabled())
    return false;

  auto &current = m_objects.top();

  const bool included = isIncludedAfterFiltering(n);

  auto type = n->type();
  switch (type) {
  case ANARI_SURFACE:
    if (m_mask & RenderInclusionMask::SURFACES) {
      size_t i = n->getObjectIndex();
      if (auto h = m_cache->getHandle(type, i, true); h != nullptr && included)
        current.surfaces.push_back((anari::Surface)h);
    }
    break;
  case ANARI_VOLUME:
    if (m_mask & RenderInclusionMask::VOLUMES) {
      size_t i = n->getObjectIndex();
      if (auto h = m_cache->getHandle(type, i, true); h != nullptr && included)
        current.volumes.push_back((anari::Volume)h);
    }
    break;
  case ANARI_LIGHT:
    if (m_mask & RenderInclusionMask::LIGHTS) {
      size_t i = n->getObjectIndex();
      if (auto h = m_cache->getHandle(type, i, true); h != nullptr)
        current.lights.push_back((anari::Light)h);
    }
    break;
  case ANARI_FLOAT32_MAT4:
    m_objects.emplace();
    break;
  case ANARI_ARRAY1D: {
    if (auto *a = n->getTransformArray(); a)
      m_objects.emplace();
  }
  default:
    break;
  }

  return true;
}

inline void RenderToAnariObjectsVisitor::postChildren(
    tsd::core::LayerNode &n, int level)
{
  if (!n->isEnabled())
    return;

  switch (n->type()) {
  case ANARI_ARRAY1D: {
    if (auto *a = n->getTransformArray(); !a)
      break;
  }
  // intentionally fallthrough...
  case ANARI_FLOAT32_MAT4: {
    if (auto inst = createInstanceFromTop()) {
      for (auto &p : n->getInstanceParameters()) {
        if (!p.second.holdsObject())
          continue;
        auto objType = p.second.type();
        auto objHandle =
            m_cache->getHandle(objType, p.second.getAsObjectIndex(), true);
        anari::setParameter(
            m_device, inst, p.first.c_str(), objType, &objHandle);
      }
      anari::commitParameters(m_device, inst);
    }
    m_objects.pop();
    break;
  }
  default:
    // no-op
    break;
  }
}

inline bool RenderToAnariObjectsVisitor::isIncludedAfterFiltering(
    const tsd::core::LayerNode &n) const
{
  if (!m_filter)
    return true;

  auto type = n->type();
  if (!anari::isObject(type))
    return false;

  return (*m_filter)(n->getObject());
}

inline anari::Instance RenderToAnariObjectsVisitor::createInstanceFromTop()
{
  auto &current = m_objects.top();
  if (current.surfaces.empty() && current.volumes.empty()
      && current.lights.empty())
    return {};

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

  current.surfaces.clear();
  current.volumes.clear();
  current.lights.clear();

  anari::commitParameters(m_device, group);

  auto instance = anari::newObject<anari::Instance>(m_device, "transform");
  anari::setParameter(m_device, instance, "group", group);
  anari::commitParameters(m_device, instance);
  m_instances->push_back(instance);

  anari::release(m_device, group);

  return instance;
}

} // namespace tsd::rendering
