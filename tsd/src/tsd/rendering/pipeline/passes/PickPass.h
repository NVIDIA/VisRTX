// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RenderPass.h"
// std
#include <functional>

namespace tsd::rendering {

struct PickPass : public RenderPass
{
  using PickOpFunc = std::function<void(RenderPass::Buffers &b)>;

  PickPass();
  ~PickPass() override;

  void setPickOperation(PickOpFunc &&f);

 private:
  void render(Buffers &b, int stageId) override;

  PickOpFunc m_op;
};

} // namespace tsd::rendering
