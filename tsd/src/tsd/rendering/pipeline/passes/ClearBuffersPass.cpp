// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ClearBuffersPass.h"
// std
#include <limits>

#ifdef ENABLE_CUDA
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#else
#include <algorithm>
#endif

namespace tsd::rendering {

ClearBuffersPass::ClearBuffersPass() = default;
ClearBuffersPass::~ClearBuffersPass() = default;

void ClearBuffersPass::setClearColor(const tsd::math::float4 &color)
{
  m_clearColor = color;
}

void ClearBuffersPass::render(ImageBuffers &b, int /*stageId*/)
{
  const auto size = getDimensions();
  const uint32_t totalPixels = uint32_t(size.x) * uint32_t(size.y);
  const auto c = helium::cvt_color_to_uint32(m_clearColor);
  const float inf = std::numeric_limits<float>::infinity();

#ifdef ENABLE_CUDA
  auto policy = thrust::cuda::par.on(b.stream);
  thrust::fill(policy, b.color, b.color + totalPixels, c);
  thrust::fill(policy, b.hdrColor, b.hdrColor + totalPixels * 4, 0.f);
  thrust::fill(policy, b.depth, b.depth + totalPixels, inf);
  thrust::fill(policy, b.objectId, b.objectId + totalPixels, ~0u);
  thrust::fill(policy, b.primitiveId, b.primitiveId + totalPixels, ~0u);
  thrust::fill(policy, b.instanceId, b.instanceId + totalPixels, ~0u);
#else
  std::fill(b.color, b.color + totalPixels, c);
  std::fill(b.hdrColor, b.hdrColor + totalPixels * 4, 0.f);
  std::fill(b.depth, b.depth + totalPixels, inf);
  std::fill(b.objectId, b.objectId + totalPixels, ~0u);
  std::fill(b.primitiveId, b.primitiveId + totalPixels, ~0u);
  std::fill(b.instanceId, b.instanceId + totalPixels, ~0u);
#endif
}

} // namespace tsd::rendering
