// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef __CUDACC__

#include "gpu/gpu_objects.h"

namespace visrtx {

// The frame compositor and camera-surface fog must use the same effective
// linear RGB, including the existing background premultiplication convention.
VISRTX_DEVICE vec4 screenBackground(
    const RendererGPUData &renderer, const vec2 &uv)
{
  vec4 bg;
  if (renderer.backgroundMode == BackgroundMode::COLOR)
    bg = renderer.background.color;
  else {
    const auto s = tex2D<float4>(renderer.background.texobj, uv.x, uv.y);
    bg = vec4(s.x, s.y, s.z, s.w);
  }
  return vec4(
      renderer.premultiplyBackground ? vec3(bg) * bg.a : vec3(bg), bg.a);
}

} // namespace visrtx

#endif // __CUDACC__
