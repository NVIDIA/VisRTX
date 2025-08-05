// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/index/RenderIndex.hpp"

namespace tsd::rendering {

struct RenderIndexFlatRegistry : public RenderIndex
{
  RenderIndexFlatRegistry(Context &ctx, anari::Device d);
  ~RenderIndexFlatRegistry() override;

  void signalObjectAdded(const Object *o) override;

 private:
  void updateWorld() override;
};

} // namespace tsd::rendering
