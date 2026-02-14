// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Instance.h"
// std
#include <vector>
// tsd_rendering
#include "tsd/rendering/index/RenderIndexAllLayers.hpp"

namespace tsd_device {

struct World : public Object
{
  World(DeviceGlobalState *s);
  ~World() override;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;

  const tsd::rendering::RenderIndexAllLayers *getRenderIndex() const;
  tsd::rendering::RenderIndexAllLayers *getRenderIndex();

 private:
  tsd::core::Layer *layer() const;

  void updateValidObjects();
  void updateLayer();

  helium::ChangeObserverPtr<helium::ObjectArray> m_zeroSurfaceData;
  helium::ChangeObserverPtr<helium::ObjectArray> m_zeroVolumeData;
  helium::ChangeObserverPtr<helium::ObjectArray> m_zeroLightData;
  helium::ChangeObserverPtr<helium::ObjectArray> m_instanceData;

  std::vector<Instance *> m_instances;
  std::vector<TSDObject *> m_zeroSurfaces;
  std::vector<TSDObject *> m_zeroVolumes;
  std::vector<TSDObject *> m_zeroLights;

  tsd::core::Token m_layerName;
  tsd::rendering::RenderIndexAllLayers *m_renderIndex{nullptr};
};

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_SPECIALIZATION(tsd_device::World *, ANARI_WORLD);
