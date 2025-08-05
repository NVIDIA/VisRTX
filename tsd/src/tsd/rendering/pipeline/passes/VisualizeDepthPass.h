// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RenderPass.h"

namespace tsd::rendering {

struct VisualizeDepthPass : public RenderPass
{
  VisualizeDepthPass();
  ~VisualizeDepthPass() override;

  void setMaxDepth(float d);

 private:
  void render(Buffers &b, int stageId) override;

  float m_maxDepth{1.f};
};

} // namespace tsd::rendering
