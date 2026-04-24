// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "PrimitiveOutlineRenderPass.h"
// tsd_algorithms
#include "tsd/algorithms/cpu/outline.hpp"
#ifdef TSD_ALGORITHMS_HAS_CUDA
#include "tsd/algorithms/cuda/outline.hpp"
#endif
// helium
#include <helium/helium_math.h>

namespace tsd::rendering {

PrimitiveOutlineRenderPass::PrimitiveOutlineRenderPass() = default;

PrimitiveOutlineRenderPass::~PrimitiveOutlineRenderPass() = default;

void PrimitiveOutlineRenderPass::setOutlineColor(
    const tsd::math::float4 &color)
{
  m_outlineColor = color;
}

void PrimitiveOutlineRenderPass::setThickness(uint32_t thickness)
{
  m_thickness = thickness;
}

void PrimitiveOutlineRenderPass::render(ImageBuffers &b, int stageId)
{
  if (!b.objectId || !b.primitiveId || stageId == 0)
    return;

  const auto size = getDimensions();
  const auto outlineColor = helium::cvt_color_to_uint32(m_outlineColor);

#ifdef TSD_ALGORITHMS_HAS_CUDA
  if (b.stream) {
    tsd::algorithms::cuda::outlinePrimitives(b.stream,
        b.objectId,
        b.primitiveId,
        b.color,
        outlineColor,
        m_thickness,
        size.x,
        size.y);
    return;
  }
#endif
  tsd::algorithms::cpu::outlinePrimitives(b.objectId,
      b.primitiveId,
      b.color,
      outlineColor,
      m_thickness,
      size.x,
      size.y);
}

} // namespace tsd::rendering
