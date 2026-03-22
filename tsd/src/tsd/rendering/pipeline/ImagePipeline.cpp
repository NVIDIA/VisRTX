// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ImagePipeline.h"
// std
#include <chrono>
#include <cstring>
#include <limits>

namespace tsd::rendering {

ImagePipeline::ImagePipeline()
{
#ifdef ENABLE_CUDA
  cudaStreamCreate(&m_buffers.stream);
#endif
}

ImagePipeline::ImagePipeline(int width, int height)
{
  setDimensions(width, height);
}

ImagePipeline::~ImagePipeline()
{
  cleanup();
#ifdef ENABLE_CUDA
  cudaStreamDestroy(m_buffers.stream);
#endif
}

void ImagePipeline::setDimensions(uint32_t width, uint32_t height)
{
  if (m_size.x == width && m_size.y == height)
    return;
  m_size.x = width;
  m_size.y = height;
  cleanup();
  const size_t totalSize = size_t(width) * size_t(height);
  m_buffers.color = detail::allocate<uint32_t>(totalSize);
  m_buffers.depth = detail::allocate<float>(totalSize);
  m_buffers.instanceId = detail::allocate<uint32_t>(totalSize);
  m_buffers.objectId = detail::allocate<uint32_t>(totalSize);
  m_buffers.primitiveId = detail::allocate<uint32_t>(totalSize);
  m_buffers.albedo = detail::allocate<tsd::math::float3>(totalSize);
  m_buffers.normal = detail::allocate<tsd::math::float3>(totalSize);
  for (auto &p : m_passes)
    p->setDimensions(width, height);
}

void ImagePipeline::render()
{
  int stageId = 0;
  m_passTimings.clear();
  for (auto &p : m_passes) {
    if (!p->isEnabled())
      continue;
    auto start = std::chrono::steady_clock::now();
    p->render(m_buffers, stageId++);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration<float, std::milli>(end - start).count();
    m_passTimings.push_back({p->name(), time});
  }
}

const uint32_t *ImagePipeline::getColorBuffer() const
{
  return m_buffers.color;
}

const std::vector<ImagePipeline::PassTiming> &ImagePipeline::getPassTimings()
    const
{
  return m_passTimings;
}

size_t ImagePipeline::size() const
{
  return m_passes.size();
}

bool ImagePipeline::empty() const
{
  return m_passes.empty();
}

void ImagePipeline::clear()
{
  m_passes.clear();
}

void ImagePipeline::cleanup()
{
  detail::free(m_buffers.color);
  detail::free(m_buffers.depth);
  detail::free(m_buffers.objectId);
  detail::free(m_buffers.primitiveId);
  detail::free(m_buffers.instanceId);
  detail::free(m_buffers.albedo);
  detail::free(m_buffers.normal);
}

} // namespace tsd::rendering
