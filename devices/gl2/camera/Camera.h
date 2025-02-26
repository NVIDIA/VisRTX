// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../Object.h"

namespace visgl2 {

// Inherit from this, add your functionality, and add it to 'createInstance()'
struct Camera : public Object
{
  Camera(VisGL2DeviceGlobalState *s);
  ~Camera() override = default;
  static Camera *createInstance(
      std::string_view type, VisGL2DeviceGlobalState *state);
};

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_SPECIALIZATION(visgl2::Camera *, ANARI_CAMERA);
