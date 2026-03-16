// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"
// std
#include <functional>

namespace tsd::rendering {

struct PickPass : public ImagePass
{
  using PickOpFunc = std::function<void(ImageBuffers &b)>;

  PickPass();
  ~PickPass() override;

  void setPickOperation(PickOpFunc &&f);

 private:
  void render(ImageBuffers &b, int stageId) override;

  PickOpFunc m_op;
};

} // namespace tsd::rendering
