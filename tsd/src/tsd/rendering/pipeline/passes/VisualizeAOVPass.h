// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"

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

/*
 * ImagePass that remaps a selected AOV channel (depth, albedo, normals,
 * edge detection, or ID buffers) into the RGBA color buffer for debugging.
 *
 * Example:
 *   auto *pass = pipeline.emplace_back<VisualizeAOVPass>();
 *   pass->setAOVType(AOVType::DEPTH);
 *   pass->setDepthRange(0.1f, 100.f);
 */
struct VisualizeAOVPass : public ImagePass
{
  VisualizeAOVPass();
  ~VisualizeAOVPass() override;
  const char *name() const override { return "Visualize AOV"; }

  void setAOVType(AOVType type);
  void setDepthRange(float minDepth, float maxDepth);
  void setEdgeThreshold(float threshold);
  void setEdgeInvert(bool invert);

 private:
  void render(ImageBuffers &b, int stageId) override;

  AOVType m_aovType{AOVType::NONE};
  float m_minDepth{0.f};
  float m_maxDepth{1.f};
  float m_edgeThreshold{0.5f};
  bool m_edgeInvert{false};
};

} // namespace tsd::rendering
