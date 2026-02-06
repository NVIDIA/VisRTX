// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RenderPass.h"

namespace tsd::rendering {

struct ClearBuffersPass : public RenderPass
{
  ClearBuffersPass();
  ~ClearBuffersPass() override;

  void setClearColor(const tsd::math::float4 &color);

 private:
  void render(RenderBuffers &b, int stageId) override;

  tsd::math::float4 m_clearColor{0.f, 0.f, 0.f, 1.f};
};

} // namespace tsd::rendering
