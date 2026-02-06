// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "MultiDeviceSceneRenderPass.h"
// tsd_core
#include <tsd/core/Logging.hpp>
// std
#include <algorithm>
#include <cstring>
#include <limits>

namespace tsd::rendering {

MultiDeviceSceneRenderPass::MultiDeviceSceneRenderPass(
    const std::vector<anari::Device> &devices)
    : m_devices(devices)
{
  for (auto d : m_devices) {
    anari::retain(d, d);
    auto f = anari::newObject<anari::Frame>(d);
    anari::setParameter(d, f, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
    anari::setParameter(d, f, "channel.depth", ANARI_FLOAT32);
    anari::setParameter(d, f, "accumulation", true);
    anari::commitParameters(d, f);
    m_frames.emplace_back(f);
  }
}

MultiDeviceSceneRenderPass::~MultiDeviceSceneRenderPass()
{
  cleanup();
  foreach_frame([](anari::Device d, anari::Frame f) {
    anari::release(d, f);
    anari::release(d, d);
  });
}

size_t MultiDeviceSceneRenderPass::numDevices() const
{
  return m_devices.size();
}

void MultiDeviceSceneRenderPass::setCamera(size_t i, anari::Camera c)
{
  auto d = m_devices[i];
  auto f = m_frames[i];
  anari::setParameter(d, f, "camera", c);
  anari::commitParameters(d, f);
}

void MultiDeviceSceneRenderPass::setRenderer(size_t i, anari::Renderer r)
{
  auto d = m_devices[i];
  auto f = m_frames[i];
  anari::setParameter(d, f, "renderer", r);
  anari::commitParameters(d, f);
}

void MultiDeviceSceneRenderPass::setWorld(size_t i, anari::World w)
{
  auto d = m_devices[i];
  auto f = m_frames[i];
  anari::setParameter(d, f, "world", w);
  anari::commitParameters(d, f);
}

void MultiDeviceSceneRenderPass::setColorFormat(anari::DataType t)
{
  foreach_frame([&](anari::Device d, anari::Frame f) {
    anari::setParameter(d, f, "channel.color", t);
    anari::commitParameters(d, f);
  });
}

void MultiDeviceSceneRenderPass::setRunAsync(bool on)
{
  m_runAsync = on;
}

void MultiDeviceSceneRenderPass::foreach_frame(
    const std::function<void(anari::Device, anari::Frame)> &func) const
{
  for (size_t i = 0; i < numDevices(); ++i)
    func(m_devices[i], m_frames[i]);
}

void MultiDeviceSceneRenderPass::updateSize()
{
  cleanup();
  auto size = getDimensions();
  foreach_frame([&](anari::Device d, anari::Frame f) {
    anari::setParameter(d, f, "size", size);
    anari::commitParameters(d, f);
  });

  const size_t totalSize = size_t(size.x) * size_t(size.y);
  m_buffers.color = detail::allocate<uint32_t>(totalSize);
  m_buffers.depth = detail::allocate<float>(totalSize);
  std::fill(m_buffers.color, m_buffers.color + totalSize, 0u);
  std::fill(m_buffers.depth,
      m_buffers.depth + totalSize,
      std::numeric_limits<float>::infinity());
}

void MultiDeviceSceneRenderPass::render(RenderBuffers &b, int stageId)
{
  m_buffers.stream = b.stream;
  foreach_frame([](anari::Device d, anari::Frame f) { anari::render(d, f); });
  foreach_frame([](anari::Device d, anari::Frame f) { anari::wait(d, f); });
  copyFrameData();
  composite(b, stageId);
}

void MultiDeviceSceneRenderPass::copyFrameData()
{
  auto d = m_devices[0];
  auto f = m_frames[0];
  auto color = anari::map<void>(d, f, "channel.color");
  auto depth = anari::map<float>(d, f, "channel.depth");

  const tsd::math::uint2 size(getDimensions());
  const size_t totalSize = size.x * size.y;
  if (totalSize > 0 && size.x == color.width && size.y == color.height) {
    if (color.pixelType == ANARI_FLOAT32_VEC4) {
      detail::convertFloatColorBuffer_(m_buffers.stream,
          (const float *)color.data,
          (uint8_t *)m_buffers.color,
          totalSize * 4);
    } else {
      detail::copy(m_buffers.color, (uint32_t *)color.data, totalSize);
    }

    detail::copy(m_buffers.depth, depth.data, totalSize);
  }

  anari::unmap(d, f, "channel.color");
  anari::unmap(d, f, "channel.depth");
}

void MultiDeviceSceneRenderPass::composite(RenderBuffers &b, int stageId)
{
  if (stageId != 0) {
    tsd::core::logWarning(
        "[MultiDeviceSceneRenderPass] "
        "pass is NOT first in the pipeline -- "
        "overriding existing frame contents");
  }

  const tsd::math::uint2 size(getDimensions());
  const size_t totalSize = size.x * size.y;

  detail::copy(b.color, m_buffers.color, totalSize);
  detail::copy(b.depth, m_buffers.depth, totalSize);
}

void MultiDeviceSceneRenderPass::cleanup()
{
  detail::free(m_buffers.color);
  detail::free(m_buffers.depth);
}

} // namespace tsd::rendering
