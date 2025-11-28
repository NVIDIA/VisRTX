// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "VisualizeAOVPass.h"
// std
#include <algorithm>

#include "detail/parallel_for.h"

namespace tsd::rendering {

// Thrust kernels /////////////////////////////////////////////////////////////

void computeDepthImage(RenderBuffers &b, float maxDepth, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        const float depth = b.depth[i];
        const float v = std::clamp(depth / maxDepth, 0.f, 1.f);
        b.color[i] = helium::cvt_color_to_uint32({tsd::math::float3(v), 1.f});
      });
}

void computeAlbedoImage(RenderBuffers &b, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        const auto albedo = b.albedo ? b.albedo[i] : tsd::math::float3(0.f);
        b.color[i] = helium::cvt_color_to_uint32({albedo, 1.f});
      });
}

void computeNormalImage(RenderBuffers &b, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        const auto normal = b.normal ? b.normal[i] : tsd::math::float3(0.f);
        // Map normals from [-1,1] to [0,1] for visualization
        const auto visualNormal = (normal + 1.f) * 0.5f;
        b.color[i] = helium::cvt_color_to_uint32({visualNormal, 1.f});
      });
}

// VisualizeAOVPass definitions ///////////////////////////////////////////////

VisualizeAOVPass::VisualizeAOVPass() = default;

VisualizeAOVPass::~VisualizeAOVPass() = default;

void VisualizeAOVPass::setAOVType(AOVType type)
{
  m_aovType = type;
  setEnabled(type != AOVType::NONE);
}

void VisualizeAOVPass::setMaxDepth(float d)
{
  m_maxDepth = d;
}

void VisualizeAOVPass::render(RenderBuffers &b, int stageId)
{
  if (stageId == 0 || m_aovType == AOVType::NONE)
    return;

  const auto size = getDimensions();

  switch (m_aovType) {
  case AOVType::DEPTH:
    computeDepthImage(b, m_maxDepth, size);
    break;
  case AOVType::ALBEDO:
    computeAlbedoImage(b, size);
    break;
  case AOVType::NORMAL:
    computeNormalImage(b, size);
    break;
  default:
    break;
  }
}

} // namespace tsd::rendering
