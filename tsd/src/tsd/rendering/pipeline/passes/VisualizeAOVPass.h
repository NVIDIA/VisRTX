// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RenderPass.h"

namespace tsd::rendering {

enum class AOVType
{
  NONE,
  DEPTH,
  ALBEDO,
  NORMAL
};

struct VisualizeAOVPass : public RenderPass
{
  VisualizeAOVPass();
  ~VisualizeAOVPass() override;

  void setAOVType(AOVType type);
  void setMaxDepth(float d);

 private:
  void render(RenderBuffers &b, int stageId) override;

  AOVType m_aovType{AOVType::NONE};
  float m_maxDepth{1.f};
};

} // namespace tsd::rendering
