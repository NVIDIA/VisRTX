// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "VisGL2DeviceGlobalState.h"
#include "VisGL2Math.h"
// helium
#include "helium/BaseObject.h"
#include "helium/utility/ChangeObserverPtr.h"
// std
#include <string_view>
// gl
#include "ogl.h"

namespace visgl2 {

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

  bool isValid() const override;

  VisGL2DeviceGlobalState *deviceState() const;

  template <typename METHOD_T>
  tasking::Future gl_enqueue_method(METHOD_T m);
};

// This gets instantiated for all object subtypes which are not known
struct UnknownObject : public Object
{
  UnknownObject(ANARIDataType type, VisGL2DeviceGlobalState *s);
  ~UnknownObject() override = default;
  bool isValid() const override;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename METHOD_T>
inline tasking::Future Object::gl_enqueue_method(METHOD_T m)
{
  auto &state = *deviceState();
  return state.gl.thread.enqueue(m, this);
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_SPECIALIZATION(visgl2::Object *, ANARI_OBJECT);
