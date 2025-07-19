// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AnariAxesRenderPass.h"
// std
#include <algorithm>
#include <cstring>
#include <limits>

#include "detail/parallel_for.h"
#include "detail/parallel_transform.h"

namespace tsd {

// AnariAxesRenderPass definitions ////////////////////////////////////////////

AnariAxesRenderPass::AnariAxesRenderPass(anari::Device d) : m_device(d)
{
  anari::retain(d, d);
  m_frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, m_frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, m_frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(d, m_frame, "accumulation", true);
}

AnariAxesRenderPass::~AnariAxesRenderPass()
{
  cleanup();

  anari::discard(m_device, m_frame);
  anari::wait(m_device, m_frame);

  anari::release(m_device, m_frame);
  anari::release(m_device, m_camera);
  anari::release(m_device, m_device);
}

void AnariAxesRenderPass::setCamera(anari::Camera c)
{
  anari::retain(m_device, c);
  anari::setParameter(m_device, m_frame, "camera", c);
  anari::commitParameters(m_device, m_frame);
  anari::release(m_device, m_camera);
  m_camera = c;
}

void AnariAxesRenderPass::updateSize()
{
  cleanup();
  auto size = getDimensions();
  anari::setParameter(m_device, m_frame, "size", size);
  anari::commitParameters(m_device, m_frame);

  const size_t totalSize = size_t(size.x) * size_t(size.y);
  m_buffers.color = detail::allocate<uint32_t>(totalSize);
  m_buffers.depth = detail::allocate<float>(totalSize);
  m_buffers.objectId = detail::allocate<uint32_t>(totalSize);
#if ENABLE_CUDA
  thrust::fill(m_buffers.color, m_buffers.color + totalSize, 0u);
  thrust::fill(m_buffers.depth,
      m_buffers.depth + totalSize,
      std::numeric_limits<float>::infinity());
  thrust::fill(m_buffers.objectId, m_buffers.objectId + totalSize, ~0u);
#else
  std::fill(m_buffers.color, m_buffers.color + totalSize, 0u);
  std::fill(m_buffers.depth,
      m_buffers.depth + totalSize,
      std::numeric_limits<float>::infinity());
  std::fill(m_buffers.objectId, m_buffers.objectId + totalSize, ~0u);
#endif
}

void AnariAxesRenderPass::render(Buffers &b, int stageId)
{
  if (m_firstFrame) {
    anari::render(m_device, m_frame);
    anari::wait(m_device, m_frame);
    m_firstFrame = false;
  }

  if (anari::isReady(m_device, m_frame)) {
    copyFrameData();
    anari::render(m_device, m_frame);
  }

  composite(b, stageId);
}

void AnariAxesRenderPass::copyFrameData()
{
#if 0
  const char *colorChannel = "channel.color";

  auto color = anari::map<void>(m_device, m_frame, colorChannel);

  const tsd::uint2 size(getDimensions());
  const size_t totalSize = size.x * size.y;
  if (totalSize > 0 && size.x == color.width && size.y == color.height) {
    if (color.pixelType == ANARI_FLOAT32_VEC4) {
      convertFloatColorBuffer(
          (const float *)color.data, (uint8_t *)m_buffers.color, totalSize * 4);
    } else
      detail::copy(m_buffers.color, (uint32_t *)color.data, totalSize);
  }

  anari::unmap(m_device, m_frame, colorChannel);
#endif
}

void AnariAxesRenderPass::composite(Buffers &b, int stageId)
{
#if 0
  const bool firstPass = stageId == 0;
  const tsd::uint2 size(getDimensions());
  const size_t totalSize = size.x * size.y;

  if (firstPass) {
    detail::copy(b.color, m_buffers.color, totalSize);
    detail::copy(b.depth, m_buffers.depth, totalSize);
    detail::copy(b.objectId, m_buffers.objectId, totalSize);
  } else {
    compositeFrame(b, m_buffers, size, firstPass);
  }
#endif
}

void AnariAxesRenderPass::cleanup()
{
  detail::free(m_buffers.color);
  detail::free(m_buffers.depth);
  detail::free(m_buffers.objectId);
}

} // namespace tsd
