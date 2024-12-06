// Copyright 2026 NVIDIA Corporation
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
  Object(anari::DataType type, DeviceGlobalState *s);
  virtual ~Object() = default;

  virtual bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;
  virtual void commitParameters() override;
  virtual void finalize() override;
  virtual bool isValid() const override;

  DeviceGlobalState *deviceState() const;
};

struct TSDObject : public Object
{
  TSDObject(anari::DataType type,
      DeviceGlobalState *s,
      tsd::core::Token subtype = tsd::core::tokens::none);
  virtual ~TSDObject() = default;

  virtual void commitParameters() override;

  tsd::core::Object *tsdObject() const;

 private:
  tsd::core::Any m_object;
};

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_SPECIALIZATION(tsd_device::TSDObject *, ANARI_OBJECT);
