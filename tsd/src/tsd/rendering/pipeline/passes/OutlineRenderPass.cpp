// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "OutlineRenderPass.h"
// tsd_algorithms
#include "tsd/algorithms/cpu/outline.hpp"
#ifdef TSD_ALGORITHMS_HAS_CUDA
#include "tsd/algorithms/cuda/outline.hpp"
#endif

namespace tsd::rendering {

// OutlineRenderPass definitions //////////////////////////////////////////////

OutlineRenderPass::OutlineRenderPass() = default;

OutlineRenderPass::~OutlineRenderPass() = default;

void OutlineRenderPass::setOutlineId(uint32_t id)
{
  m_outlineId = id;
}

void OutlineRenderPass::render(ImageBuffers &b, int stageId)
{
  if (!b.objectId || stageId == 0 || m_outlineId == ~0u)
    return;

  const auto size = getDimensions();

#ifdef TSD_ALGORITHMS_HAS_CUDA
  if (b.stream) {
    tsd::algorithms::cuda::outlineObject(
        b.stream, b.objectId, b.color, m_outlineId, size.x, size.y);
    return;
  }
#endif
  tsd::algorithms::cpu::outlineObject(
      b.objectId, b.color, m_outlineId, size.x, size.y);
}

} // namespace tsd::rendering
