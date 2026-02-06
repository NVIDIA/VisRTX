// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ClearBuffersPass.h"
// std
#include <algorithm>
#include <limits>

#include "detail/parallel_for.h"

namespace tsd::rendering {

ClearBuffersPass::ClearBuffersPass() = default;
ClearBuffersPass::~ClearBuffersPass() = default;

void ClearBuffersPass::setClearColor(const tsd::math::float4 &color)
{
  m_clearColor = color;
}

void ClearBuffersPass::render(RenderBuffers &b, int /*stageId*/)
{
  const auto size = getDimensions();
  const size_t totalSize = size_t(size.x) * size_t(size.y);
  const auto c = helium::cvt_color_to_uint32(m_clearColor);
#if ENABLE_CUDA
  thrust::fill(b.color, b.color + totalSize, c);
  thrust::fill(b.depth, b.depth + totalSize, tsd::core::math::inf);
  thrust::fill(b.objectId, b.objectId + totalSize, ~0u);
#else
  std::fill(b.color, b.color + totalSize, c);
  std::fill(b.depth, b.depth + totalSize, tsd::core::math::inf);
  std::fill(b.objectId, b.objectId + totalSize, ~0u);
#endif
}

} // namespace tsd::rendering
