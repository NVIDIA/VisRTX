// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "Object.h"

namespace visgl2 {

// Object definitions /////////////////////////////////////////////////////////

Object::Object(ANARIDataType type, VisGL2DeviceGlobalState *s)
    : helium::BaseObject(type, s)
{}

void Object::commitParameters()
{
  // no-op
}

void Object::finalize()
{
  // no-op
}

bool Object::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (name == "valid" && type == ANARI_BOOL) {
    helium::writeToVoidP(ptr, isValid());
    return true;
  }

  return false;
}

bool Object::isValid() const
{
  return true;
}

VisGL2DeviceGlobalState *Object::deviceState() const
{
  return (VisGL2DeviceGlobalState *)helium::BaseObject::m_state;
}

// UnknownObject definitions //////////////////////////////////////////////////

UnknownObject::UnknownObject(ANARIDataType type, VisGL2DeviceGlobalState *s)
    : Object(type, s)
{}

bool UnknownObject::isValid() const
{
  return false;
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_DEFINITION(visgl2::Object *);
