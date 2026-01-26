// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CopyFromColorBufferPass.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// std
#include <cstring>

namespace tsd::rendering {

CopyFromColorBufferPass::CopyFromColorBufferPass() = default;

CopyFromColorBufferPass::~CopyFromColorBufferPass() = default;

void CopyFromColorBufferPass::setExternalBuffer(std::vector<uint8_t> &buffer)
{
  m_externalBuffer = &buffer;
}

void CopyFromColorBufferPass::render(RenderBuffers &b, int /*stageId*/)
{
  if (!b.color) {
    tsd::core::logError("[CopyFromColorBufferPass] No color buffer available");
    return;
  }

  if (!m_externalBuffer) {
    tsd::core::logError("[CopyFromColorBufferPass] No external buffer set");
    return;
  }

  const auto size = getDimensions();
  const size_t totalPixels = size.x * size.y;

  if (totalPixels == 0) {
    tsd::core::logError("[CopyFromColorBufferPass] Invalid dimensions");
    return;
  }

  m_externalBuffer->resize(totalPixels * 4); // RGBA8
  std::memcpy(m_externalBuffer->data(), b.color, totalPixels * 4);
}

} // namespace tsd::rendering
