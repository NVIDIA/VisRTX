// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "Camera.h"

namespace visgl2 {

Camera::Camera(VisGL2DeviceGlobalState *s) : Object(ANARI_CAMERA, s) {}

Camera *Camera::createInstance(
    std::string_view /*type*/, VisGL2DeviceGlobalState *s)
{
  return (Camera *)new UnknownObject(ANARI_CAMERA, s);
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_DEFINITION(visgl2::Camera *);
