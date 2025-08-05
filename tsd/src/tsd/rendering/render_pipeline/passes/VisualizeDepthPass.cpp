// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "VisualizeDepthPass.h"
// std
#include <algorithm>

#include "detail/parallel_for.h"

namespace tsd::rendering {

// Thrust kernels /////////////////////////////////////////////////////////////

void computeDepthImage(
    RenderPass::Buffers &b, float maxDepth, tsd::math::uint2 size)
{
  detail::parallel_for(
      0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        const float depth = b.depth[i];
        const float v = std::clamp(depth / maxDepth, 0.f, 1.f);
        b.color[i] = helium::cvt_color_to_uint32({tsd::math::float3(v), 1.f});
      });
}

// VisualizeDepthPass definitions /////////////////////////////////////////////

VisualizeDepthPass::VisualizeDepthPass() = default;

VisualizeDepthPass::~VisualizeDepthPass() = default;

void VisualizeDepthPass::setMaxDepth(float d)
{
  m_maxDepth = d;
}

void VisualizeDepthPass::render(Buffers &b, int stageId)
{
  if (stageId == 0)
    return;

  computeDepthImage(b, m_maxDepth, getDimensions());
}

} // namespace tsd::rendering
