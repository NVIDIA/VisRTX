// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"
// std
#include <string>

namespace tsd::rendering {

/*
 * ImagePass that writes the current color buffer to an image file; in
 * single-shot mode the pass disables itself after the first successful write.
 *
 * Example:
 *   auto *pass = pipeline.emplace_back<SaveToFilePass>();
 *   pass->setFilename("frame.png");
 *   pass->setSingleShotMode(true);
 */
struct SaveToFilePass : public ImagePass
{
  SaveToFilePass();
  ~SaveToFilePass() override;
  const char *name() const override { return "Save To File"; }

  void setFilename(const std::string &filename);
  const std::string &getFilename() const;

  void setSingleShotMode(bool enabled);

 private:
  void render(ImageBuffers &b, int stageId) override;

  std::string m_filename;
  bool m_singleShot{true};
};

} // namespace tsd::rendering
