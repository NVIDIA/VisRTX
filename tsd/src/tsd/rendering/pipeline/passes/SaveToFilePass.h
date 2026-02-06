// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RenderPass.h"
// std
#include <string>

namespace tsd::rendering {

struct SaveToFilePass : public RenderPass
{
  SaveToFilePass();
  ~SaveToFilePass() override;

  void setFilename(const std::string &filename);
  const std::string &getFilename() const;

  void setSingleShotMode(bool enabled);

 private:
  void render(RenderBuffers &b, int stageId) override;

  std::string m_filename;
  bool m_singleShot{true};
};

} // namespace tsd::rendering
