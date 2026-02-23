// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <helium/utility/TimeStamp.h>
#include "tsd/core/scene/Layer.hpp"
#include "tsd/rendering/index/RenderIndex.hpp"
// std
#include <anari/anari_cpp.hpp>
#include <variant>
#include <vector>

namespace tsd::rendering {

struct RenderIndexAllLayers : public RenderIndex
{
  RenderIndexAllLayers(
      Scene &scene, tsd::core::Token deviceName, anari::Device d);
  ~RenderIndexAllLayers() override;

  bool isFlat() const override;

  void setFilterFunction(RenderIndexFilterFcn f) override;

  void setIncludedLayers(const std::vector<const Layer *> &layers);

  void signalArrayUnmapped(const Array *a) override;
  void signalParameterUpdated(const Object *o, const Parameter *p) override;
  void signalObjectLayerUseCountZero(const Object *obj) override;
  void signalLayerAdded(const Layer *l) override;
  void signalLayerUpdated(const Layer *l) override;
  void signalLayerRemoved(const Layer *l) override;
  void signalActiveLayersChanged() override;
  void signalObjectFilteringChanged() override;
  void signalAnimationTimeChanged(float time) override;
  void signalRemoveAllObjects() override;

 private:
  void updateWorld() override;

  RenderIndexFilterFcn m_filter;
  std::vector<const Layer *> m_includedLayers;
  bool m_customIncludedLayers{false};
  bool m_filterForceUpdate{false};

  void tagDirtyTopology(const Layer *l)
  {
    m_layerTransformDependency.erase(l);
    m_layerTransformCache.erase(l);
    if (auto *instances = m_layerInstanceCache.at(l)) {
      for (auto &inst : *instances)
        anari::release(device(), inst);
    }
    m_layerInstanceCache.erase(l);
    if (auto *rootInst = m_layerRootInstance.at(l)) {
      anari::release(device(), *rootInst);
      m_layerRootInstance.erase(l);
    }
  }

  helium::TimeStamp m_transformLastUpdateTimeStamp{0};

  struct TransformDependency
  {
    size_t transformObjectIndex;
    size_t parentOrderedIndex;
    std::string transformName;
  };

  struct TransformCache
  {
    using Value = std::variant<math::mat4, std::vector<math::mat4>>;
    Value value;
    helium::TimeStamp timestamp{};
  };

  FlatMap<const Layer *, std::vector<TransformDependency>>
      m_layerTransformDependency;
  FlatMap<const Layer *, std::vector<TransformCache>> m_layerTransformCache;
  FlatMap<const Layer *, std::vector<anari::Instance>> m_layerInstanceCache;
  FlatMap<const Layer *, anari::Instance> m_layerRootInstance;

  void updateLayerTransformDependency(const Layer *l);
  bool updateLayerTransformCache(const Layer *l);
  void writeTransformCacheToInstances();
  // void patchInstanceTransforms();
  bool invalidateTransformAtObjectIndex(size_t index);
  bool invalidateTransformAtObjectIndex(size_t index, const Layer *l);
};

} // namespace tsd::rendering
