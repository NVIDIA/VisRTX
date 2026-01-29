// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RenderPass.h"

namespace tsd::rendering {

enum class AOVType
{
  NONE,
  DEPTH,
  ALBEDO,
  NORMAL,
  EDGES,
  OBJECT_ID,
  PRIMITIVE_ID,
  INSTANCE_ID
};

struct VisualizeAOVPass : public RenderPass
{
  VisualizeAOVPass();
  ~VisualizeAOVPass() override;

  void setAOVType(AOVType type);
  void setDepthRange(float minDepth, float maxDepth);
  void setEdgeThreshold(float threshold);
  void setEdgeInvert(bool invert);

 private:
  void render(RenderBuffers &b, int stageId) override;

  AOVType m_aovType{AOVType::NONE};
  float m_minDepth{0.f};
  float m_maxDepth{1.f};
  float m_edgeThreshold{0.5f};
  bool m_edgeInvert{false};
};

} // namespace tsd::rendering
