// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
// helium
#include <helium/array/ObjectArray.h>

namespace tsd_device {

struct Group : public Object
{
  Group(DeviceGlobalState *s);
  virtual ~Group() = default;

  void commitParameters() override;
  void finalize() override;

  const std::vector<TSDObject *> &surfaces() const;
  const std::vector<TSDObject *> &volumes() const;
  const std::vector<TSDObject *> &lights() const;

  void addObjectsToLayer(tsd::core::LayerNodeRef parent) const;

 private:
  helium::ChangeObserverPtr<helium::ObjectArray> m_surfaceData;
  helium::ChangeObserverPtr<helium::ObjectArray> m_volumeData;
  helium::ChangeObserverPtr<helium::ObjectArray> m_lightData;

  std::vector<TSDObject *> m_surfaces;
  std::vector<TSDObject *> m_volumes;
  std::vector<TSDObject *> m_lights;
};

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_SPECIALIZATION(tsd_device::Group *, ANARI_GROUP);
