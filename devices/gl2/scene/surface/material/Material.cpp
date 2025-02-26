// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "Material.h"

namespace visgl2 {

Material::Material(VisGL2DeviceGlobalState *s) : Object(ANARI_MATERIAL, s) {}

Material *Material::createInstance(
    std::string_view subtype, VisGL2DeviceGlobalState *s)
{
  return (Material *)new UnknownObject(ANARI_MATERIAL, s);
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_DEFINITION(visgl2::Material *);
