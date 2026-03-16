// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"
// std
#include <string>

namespace tsd::rendering {

struct SaveToFilePass : public ImagePass
{
  SaveToFilePass();
  ~SaveToFilePass() override;

  void setFilename(const std::string &filename);
  const std::string &getFilename() const;

  void setSingleShotMode(bool enabled);

 private:
  void render(ImageBuffers &b, int stageId) override;

  std::string m_filename;
  bool m_singleShot{true};
};

} // namespace tsd::rendering
