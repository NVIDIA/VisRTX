// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/index/RenderIndex.hpp"
// std
#include <vector>

namespace tsd::rendering {

struct RenderIndexAllLayers : public RenderIndex
{
  RenderIndexAllLayers(
      Scene &scene, anari::Device d, bool alwaysGatherAllLights = false);
  ~RenderIndexAllLayers() override;

  bool isFlat() const override;

  void setFilterFunction(RenderIndexFilterFcn f) override;

  void setIncludedLayers(const std::vector<const Layer *> &layers);

  void signalArrayUnmapped(const Array *a) override;
  void signalObjectParameterUseCountZero(const Object *obj) override;
  void signalObjectLayerUseCountZero(const Object *obj) override;
  void signalLayerAdded(const Layer *l) override;
  void signalLayerUpdated(const Layer *l) override;
  void signalLayerRemoved(const Layer *l) override;
  void signalActiveLayersChanged() override;
  void signalObjectFilteringChanged() override;
  void signalRemoveAllObjects() override;

 private:
  void updateWorld() override;
  void syncLayerInstances(
      const Layer *layer, bool appendExisting, uint8_t mask);
  void releaseAllInstances();

  RenderIndexFilterFcn m_filter;
  std::vector<const Layer *> m_includedLayers;
  bool m_forceAllLights{false};
  bool m_customIncludedLayers{false};
  bool m_filterForceUpdate{false};

  using InstanceCache = FlatMap<const Layer *, std::vector<anari::Instance>>;
  InstanceCache m_instanceCache;
};

} // namespace tsd::rendering
