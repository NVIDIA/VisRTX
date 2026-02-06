// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CopyToColorBufferPass.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// std
#include <cstring>

namespace tsd::rendering {

CopyToColorBufferPass::CopyToColorBufferPass() = default;

CopyToColorBufferPass::~CopyToColorBufferPass() = default;

void CopyToColorBufferPass::setExternalBuffer(std::vector<uint8_t> &buffer)
{
  m_externalBuffer = &buffer;
}

void CopyToColorBufferPass::render(RenderBuffers &b, int /*stageId*/)
{
  if (!b.color) {
    tsd::core::logError("[CopyToColorBufferPass] No color buffer available");
    return;
  }

  if (!m_externalBuffer) {
    tsd::core::logError("[CopyToColorBufferPass] No external buffer set");
    return;
  }

  const auto size = getDimensions();
  const size_t totalPixels = size.x * size.y;

  if (totalPixels != m_externalBuffer->size() / 4) {
    tsd::core::logError(
        "[CopyToColorBufferPass] Mismatched dimensions, skipping copy");
    return;
  }

  std::memcpy(b.color, m_externalBuffer->data(), totalPixels * 4);
}

} // namespace tsd::rendering
