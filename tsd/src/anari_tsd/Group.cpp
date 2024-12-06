// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Group.h"
// std
#include <algorithm>
#include <iterator>

namespace tsd_device {

Group::Group(DeviceGlobalState *s)
    : Object(ANARI_GROUP, s),
      m_surfaceData(this),
      m_volumeData(this),
      m_lightData(this)
{}

void Group::commitParameters()
{
  m_surfaceData = getParamObject<helium::ObjectArray>("surface");
  m_volumeData = getParamObject<helium::ObjectArray>("volume");
  m_lightData = getParamObject<helium::ObjectArray>("light");
}

void Group::finalize()
{
  m_surfaces.clear();
  m_volumes.clear();
  m_lights.clear();

  if (m_surfaceData) {
    std::transform(m_surfaceData->handlesBegin(),
        m_surfaceData->handlesEnd(),
        std::back_inserter(m_surfaces),
        [](auto *o) { return (TSDObject *)o; });
  }
  if (m_volumeData) {
    std::transform(m_volumeData->handlesBegin(),
        m_volumeData->handlesEnd(),
        std::back_inserter(m_volumes),
        [](auto *o) { return (TSDObject *)o; });
  }
  if (m_lightData) {
    std::transform(m_lightData->handlesBegin(),
        m_lightData->handlesEnd(),
        std::back_inserter(m_lights),
        [](auto *o) { return (TSDObject *)o; });
  }
}

const std::vector<TSDObject *> &Group::surfaces() const
{
  return m_surfaces;
}

const std::vector<TSDObject *> &Group::volumes() const
{
  return m_volumes;
}

const std::vector<TSDObject *> &Group::lights() const
{
  return m_lights;
}

void Group::addObjectsToLayer(tsd::core::LayerNodeRef parent) const
{
  for (auto *obj : m_surfaces)
    parent->insert_last_child(obj->tsdObject());
  for (auto *obj : m_volumes)
    parent->insert_last_child(obj->tsdObject());
  for (auto *obj : m_lights)
    parent->insert_last_child(obj->tsdObject());
}

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_DEFINITION(tsd_device::Group *);
