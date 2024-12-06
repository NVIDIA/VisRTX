// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceGlobalState.h"
// helium
#include <helium/BaseObject.h>
#include <helium/utility/ChangeObserverPtr.h>
// std
#include <string_view>

namespace tsd_device {

struct Object : public helium::BaseObject
{
  Object(anari::DataType type,
      DeviceGlobalState *s,
      tsd::Token subtype = tsd::tokens::none);
  virtual ~Object() = default;

  virtual bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;
  virtual void commitParameters() override;
  virtual void finalize() override;
  bool isValid() const override;

  DeviceGlobalState *deviceState() const;

  tsd::Object *tsdObject() const;

 private:
  tsd::Object *m_object{nullptr};
};

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_SPECIALIZATION(tsd_device::Object *, ANARI_OBJECT);
