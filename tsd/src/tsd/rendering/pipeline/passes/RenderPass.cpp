// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderPass.h"
// std
#include <algorithm>
#include <cstring>

#if USE_CUDA
#include "cuda_runtime.h"
#endif

#include "detail/parallel_transform.h"

namespace tsd::rendering {

RenderPass::RenderPass() = default;

RenderPass::~RenderPass() = default;

void RenderPass::setEnabled(bool enabled)
{
  m_enabled = enabled;
}

bool RenderPass::isEnabled() const
{
  return m_enabled;
}

void RenderPass::updateSize()
{
  // no-up
}

tsd::math::uint2 RenderPass::getDimensions() const
{
  return m_size;
}

void RenderPass::setDimensions(uint32_t width, uint32_t height)
{
  if (m_size.x == width && m_size.y == height)
    return;

  m_size.x = width;
  m_size.y = height;

  updateSize();
}

// Utility functions //////////////////////////////////////////////////////////

namespace detail {

void *allocate_(size_t numBytes)
{
#ifdef ENABLE_CUDA
  void *ptr = nullptr;
  cudaMallocManaged(&ptr, numBytes);
  return ptr;
#else
  return std::malloc(numBytes);
#endif
}

void free_(void *ptr)
{
#ifdef ENABLE_CUDA
  cudaFree(ptr);
#else
  std::free(ptr);
#endif
}

void memcpy_(void *dst, const void *src, size_t numBytes)
{
#ifdef ENABLE_CUDA
  cudaMemcpy(dst, src, numBytes, cudaMemcpyDefault);
#else
  std::memcpy(dst, src, numBytes);
#endif
}

void convertFloatColorBuffer_(
    ComputeStream stream, const float *v, uint8_t *out, size_t totalSize)
{
  detail::parallel_transform(
      stream, v, v + totalSize, out, [] DEVICE_FCN(float v) {
        return uint8_t(std::clamp(v, 0.f, 1.f) * 255);
      });
}

} // namespace detail

} // namespace tsd::rendering
