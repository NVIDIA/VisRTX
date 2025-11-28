// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderPipeline.h"
// std
#include <chrono>
#include <cstring>
#include <limits>

namespace tsd::rendering {

RenderPipeline::RenderPipeline()
{
#ifdef ENABLE_CUDA
  cudaStreamCreate(&m_buffers.stream);
#endif
}

RenderPipeline::RenderPipeline(int width, int height)
{
  setDimensions(width, height);
}

RenderPipeline::~RenderPipeline()
{
  void cleanup();
#ifdef ENABLE_CUDA
  cudaStreamDestroy(m_buffers.stream);
#endif
}

void RenderPipeline::setDimensions(uint32_t width, uint32_t height)
{
  if (m_size.x == width && m_size.y == height)
    return;
  m_size.x = width;
  m_size.y = height;
  cleanup();
  const size_t totalSize = size_t(width) * size_t(height);
  m_buffers.color = detail::allocate<uint32_t>(totalSize);
  m_buffers.depth = detail::allocate<float>(totalSize);
  m_buffers.objectId = detail::allocate<uint32_t>(totalSize);
  m_buffers.albedo = detail::allocate<tsd::math::float3>(totalSize);
  m_buffers.normal = detail::allocate<tsd::math::float3>(totalSize);
  for (auto &p : m_passes)
    p->setDimensions(width, height);
}

void RenderPipeline::render()
{
  int stageId = 0;
#define PRINT_LATENCIES 0
  for (auto &p : m_passes) {
    if (!p->isEnabled())
      continue;
#if PRINT_LATENCIES
    auto start = std::chrono::steady_clock::now();
#endif
    p->render(m_buffers, stageId++);
#if PRINT_LATENCIES
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration<float>(end - start).count();
    printf("[%i] %fms\t", stageId - 1, time * 1000);
#endif
  }
#if PRINT_LATENCIES
  printf("\n");
#endif
}

const uint32_t *RenderPipeline::getColorBuffer() const
{
  return m_buffers.color;
}

size_t RenderPipeline::size() const
{
  return m_passes.size();
}

bool RenderPipeline::empty() const
{
  return m_passes.empty();
}

void RenderPipeline::clear()
{
  m_passes.clear();
}

void RenderPipeline::cleanup()
{
  detail::free(m_buffers.color);
  detail::free(m_buffers.depth);
  detail::free(m_buffers.objectId);
  detail::free(m_buffers.albedo);
  detail::free(m_buffers.normal);
}

} // namespace tsd::rendering
