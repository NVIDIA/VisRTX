// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "gpu/gpu_objects.h"

namespace visrtx {
// UV is the rectangle's edge1/edge2 coordinate on both estimator paths.
// Sampling stays uniform in area, so texture modulation changes radiance only.
VISRTX_DEVICE vec3 quadLightTexture(
    const RectLightGPUData &rect, const vec2 &uv)
{
  if (!rect.colorMap)
    return vec3(1.f);
  const float4 value = tex2D<float4>(rect.colorMap, uv.x, uv.y);
  return vec3(value.x, value.y, value.z);
}
} // namespace visrtx
