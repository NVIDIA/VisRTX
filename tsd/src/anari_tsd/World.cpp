// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "World.h"

namespace tsd_device {

World::World(DeviceGlobalState *s)
    : Object(ANARI_WORLD, s),
      m_zeroSurfaceData(this),
      m_zeroVolumeData(this),
      m_zeroLightData(this),
      m_instanceData(this)
{
  m_layerName = "world" + std::to_string(s->worldCount++);
  auto *l = s->scene.addLayer(m_layerName);
  m_renderIndex =
      (tsd::rendering::RenderIndexAllLayers *)s->anari.acquireRenderIndex(
          s->scene, s->device);
}

World::~World()
{
  auto *s = deviceState();
  s->scene.removeLayer(m_layerName);
}

bool World::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t size,
    uint32_t flags)
{
  std::string nameStr(name);
  return anariGetProperty(deviceState()->device,
      m_renderIndex->world(),
      nameStr.c_str(),
      type,
      ptr,
      size,
      flags);
}

void World::commitParameters()
{
  m_zeroSurfaceData = getParamObject<helium::ObjectArray>("surface");
  m_zeroVolumeData = getParamObject<helium::ObjectArray>("volume");
  m_zeroLightData = getParamObject<helium::ObjectArray>("light");
  m_instanceData = getParamObject<helium::ObjectArray>("instance");
}

void World::finalize()
{
  updateValidObjects();
  updateLayer();
}

const tsd::rendering::RenderIndexAllLayers *World::getRenderIndex() const
{
  return m_renderIndex;
}

tsd::core::Layer *World::layer() const
{
  auto *s = deviceState();
  return s->scene.layer(m_layerName);
}

void World::updateValidObjects()
{
  m_instances.clear();
  if (m_instanceData) {
    std::for_each(m_instanceData->handlesBegin(),
        m_instanceData->handlesEnd(),
        [&](auto *o) {
          if (o)
            m_instances.push_back((Instance *)o);
        });
  }

  m_zeroSurfaces.clear();
  if (m_zeroSurfaceData) {
    std::for_each(m_zeroSurfaceData->handlesBegin(),
        m_zeroSurfaceData->handlesEnd(),
        [&](auto *o) {
          if (o)
            m_zeroSurfaces.push_back((TSDObject *)o);
        });
  }

  m_zeroVolumes.clear();
  if (m_zeroVolumeData) {
    std::for_each(m_zeroVolumeData->handlesBegin(),
        m_zeroVolumeData->handlesEnd(),
        [&](auto *o) {
          if (o)
            m_zeroVolumes.push_back((TSDObject *)o);
        });
  }

  m_zeroLights.clear();
  if (m_zeroLightData) {
    std::for_each(m_zeroLightData->handlesBegin(),
        m_zeroLightData->handlesEnd(),
        [&](auto *o) {
          if (o)
            m_zeroLights.push_back((TSDObject *)o);
        });
  }
}

void World::updateLayer()
{
  auto *l = layer();
  l->erase_subtree(l->root());

  for (auto *inst : m_instances) {
    auto instNode = l->root()->insert_last_child(inst->xfm());
    inst->group()->addObjectsToLayer(instNode);
  }

  if (m_zeroSurfaceData || m_zeroVolumeData || m_zeroLightData) {
    auto zi = l->root()->insert_last_child("zero-instance");
    for (auto *obj : m_zeroSurfaces)
      zi->insert_last_child(obj->tsdObject());
    for (auto *obj : m_zeroVolumes)
      zi->insert_last_child(obj->tsdObject());
    for (auto *obj : m_zeroLights)
      zi->insert_last_child(obj->tsdObject());
  }

  deviceState()->scene.signalLayerChange(l);
}

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_DEFINITION(tsd_device::World *);
