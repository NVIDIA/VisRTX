// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "Light.h"

namespace visgl2 {

Light::Light(VisGL2DeviceGlobalState *s) : Object(ANARI_LIGHT, s) {}

Light *Light::createInstance(std::string_view /*subtype*/, VisGL2DeviceGlobalState *s)
{
  return (Light *)new UnknownObject(ANARI_LIGHT, s);
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_DEFINITION(visgl2::Light *);
