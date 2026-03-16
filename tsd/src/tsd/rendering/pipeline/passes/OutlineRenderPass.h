// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"

namespace tsd::rendering {

struct OutlineRenderPass : public ImagePass
{
  OutlineRenderPass();
  ~OutlineRenderPass() override;

  void setOutlineId(uint32_t id);

 private:
  void render(ImageBuffers &b, int stageId) override;

  uint32_t m_outlineId{~0u};
};

} // namespace tsd::rendering
