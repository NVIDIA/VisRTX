// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"
// tsd_core
#include "tsd/core/TSDMath.hpp"

namespace tsd::rendering {

struct PrimitiveOutlineRenderPass : public ImagePass
{
  PrimitiveOutlineRenderPass();
  ~PrimitiveOutlineRenderPass() override;
  const char *name() const override { return "Primitive Outline"; }

  void setOutlineColor(const tsd::math::float4 &color);
  void setThickness(uint32_t thickness);

 private:
  void render(ImageBuffers &b, int stageId) override;

  tsd::math::float4 m_outlineColor{0.8f, 0.8f, 0.8f, 1.f};
  uint32_t m_thickness{1};
};

} // namespace tsd::rendering
