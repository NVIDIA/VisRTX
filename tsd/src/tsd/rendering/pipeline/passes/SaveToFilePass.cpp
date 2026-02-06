// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "SaveToFilePass.h"
#include "tsd/core/Logging.hpp"
// stb_image
#include "stb_image_write.h"

namespace tsd::rendering {

SaveToFilePass::SaveToFilePass() = default;

SaveToFilePass::~SaveToFilePass() = default;

void SaveToFilePass::setFilename(const std::string &filename)
{
  m_filename = filename;
}

const std::string &SaveToFilePass::getFilename() const
{
  return m_filename;
}

void SaveToFilePass::setSingleShotMode(bool enabled)
{
  m_singleShot = enabled;
}

void SaveToFilePass::render(RenderBuffers &b, int /*stageId*/)
{
  if (m_filename.empty()) {
    tsd::core::logWarning("[SaveToFilePass] No filename set, skipping save");
    return;
  }

  if (!b.color) {
    tsd::core::logError("[SaveToFilePass] No color buffer available");
    return;
  }

  const auto size = getDimensions();
  const size_t totalPixels = size.x * size.y;

  if (totalPixels == 0) {
    tsd::core::logError("[SaveToFilePass] Invalid dimensions");
    return;
  }

  // Use STB's built-in vertical flip functionality for simplicity
  stbi_flip_vertically_on_write(1);

  // Write PNG file (4 components = RGBA, stride = width * 4 bytes)
  int result = stbi_write_png(m_filename.c_str(),
      static_cast<int>(size.x),
      static_cast<int>(size.y),
      4, // RGBA
      b.color,
      static_cast<int>(size.x) * 4);

  if (result) {
    tsd::core::logStatus(
        "[SaveToFilePass] Saved image to '%s'", m_filename.c_str());
  } else {
    tsd::core::logWarning(
        "[SaveToFilePass] Failed to save image to '%s'", m_filename.c_str());
  }

  if (m_singleShot)
    setEnabled(false);
}

} // namespace tsd::rendering
