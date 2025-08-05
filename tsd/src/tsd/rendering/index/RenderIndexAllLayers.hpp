// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/index/RenderIndex.hpp"
// std
#include <vector>

namespace tsd::rendering {

using InstanceCache = FlatMap<const Layer *, std::vector<anari::Instance>>;

struct RenderIndexAllLayers : public RenderIndex
{
  RenderIndexAllLayers(Context &ctx, anari::Device d);
  ~RenderIndexAllLayers() override;

  void setFilterFunction(RenderIndexFilterFcn f) override;

  void signalArrayUnmapped(const Array *a) override;
  void signalLayerAdded(const Layer *l) override;
  void signalLayerUpdated(const Layer *l) override;
  void signalLayerRemoved(const Layer *l) override;
  void signalObjectFilteringChanged() override;
  void signalRemoveAllObjects() override;

 private:
  void updateWorld() override;
  void syncLayerInstances(const Layer *layer);
  void releaseAllInstances();

  RenderIndexFilterFcn m_filter;
  bool m_filterForceUpdate{false};
  InstanceCache m_instanceCache;
};

} // namespace tsd::rendering
