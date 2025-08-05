// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RenderPass.h"

namespace tsd::rendering {

struct OutlineRenderPass : public RenderPass
{
  OutlineRenderPass();
  ~OutlineRenderPass() override;

  void setOutlineId(uint32_t id);

 private:
  void render(Buffers &b, int stageId) override;

  uint32_t m_outlineId{~0u};
};

} // namespace tsd::rendering
