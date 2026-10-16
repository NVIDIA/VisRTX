// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../Object.h"
// std
#include <optional>

namespace halcyon {

struct Camera : public Object
{
  Camera(HalcyonDeviceGlobalState *s);
  ~Camera() override;

  static Camera *createInstance(
      std::string_view type, HalcyonDeviceGlobalState *state);

  virtual void commitParameters() override;
  virtual void finalize() override;

 protected:
  float3 m_pos;
  float3 m_dir;
  float3 m_up;
  float4 m_imageRegion;
  // unset uses the frame's aspect ratio, as in helide
  std::optional<float> m_aspect;
};

} // namespace halcyon

HALCYON_ANARI_TYPEFOR_SPECIALIZATION(halcyon::Camera *, ANARI_CAMERA);
