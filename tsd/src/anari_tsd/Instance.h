// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Group.h"

namespace tsd_device {

struct Instance : public Object
{
  Instance(DeviceGlobalState *s, tsd::core::Token subtype);
  virtual ~Instance() = default;

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  uint32_t numTransforms() const;

  const anari::math::mat4 &xfm(uint32_t i = 0) const;

  const Group *group() const;

 private:
  anari::math::mat4 m_xfm;
  std::vector<anari::math::mat4> m_invXfmData;
  helium::ChangeObserverPtr<helium::Array1D> m_xfmArray;
  helium::IntrusivePtr<Group> m_group;
};

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_SPECIALIZATION(tsd_device::Instance *, ANARI_INSTANCE);
