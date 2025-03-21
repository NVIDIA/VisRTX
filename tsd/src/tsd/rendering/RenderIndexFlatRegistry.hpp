// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/RenderIndex.hpp"

namespace tsd {

struct RenderIndexFlatRegistry : public RenderIndex
{
  RenderIndexFlatRegistry(anari::Device d);
  ~RenderIndexFlatRegistry() override;

 private:
  void updateWorld() override;
};

} // namespace tsd
