// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/RenderIndex.hpp"

namespace tsd {

struct RenderIndexTreeHierarchy : public RenderIndex
{
  RenderIndexTreeHierarchy(anari::Device d);
  ~RenderIndexTreeHierarchy() override;

  void setFilterFunction(RenderIndexFilterFcn f) override;

  void signalArrayUnmapped(const Array *a) override;
  void signalLayerChanged() override;
  void signalObjectFilteringChanged() override;

 private:
  void updateWorld() override;

  RenderIndexFilterFcn m_filter;
};

} // namespace tsd
