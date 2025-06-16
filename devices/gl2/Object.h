// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "VisGL2DeviceGlobalState.h"
#include "VisGL2Math.h"
// helium
#include "helium/BaseObject.h"
#include "helium/utility/ChangeObserverPtr.h"
// std
#include <string_view>

namespace visgl2 {

// This is the base of all objects other than arrays and frames, where all
// truely device-wide generic object data and behaviors can be consolidated. The
// main addition this has over helium::BaseObject is the concept of "object
// validity" (isValid() virtual function).
struct Object : public helium::BaseObject
{
  Object(ANARIDataType type, VisGL2DeviceGlobalState *s);
  virtual ~Object() = default;

  virtual bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;

  virtual void commitParameters() override;
  virtual void finalize() override;

  // Returns whether this object is valid or not. Validity is defined as being
  // a known object subtype and it is sufficiently parameterized to do something
  // useful in rendering a frame. When assembling objects in the scene behind
  // the API, it is useful to skip invalid objects as invalid/unknown object
  // usage should not be fatal to the application.
  bool isValid() const override;

  // Convenience function to get the generic helium::DeviceGlobalState as this
  // device's specific global state type.
  VisGL2DeviceGlobalState *deviceState() const;
};

// This gets instantiated for all object subtypes which are not known
struct UnknownObject : public Object
{
  UnknownObject(ANARIDataType type, VisGL2DeviceGlobalState *s);
  ~UnknownObject() override = default;
  bool isValid() const override;
};

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_SPECIALIZATION(visgl2::Object *, ANARI_OBJECT);
