// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/rendering/index/RenderIndexAllLayers.hpp"

#include "RenderToAnariObjectsVisitor.hpp"
// std
#include <algorithm>

namespace tsd::rendering {

// Helper functions ///////////////////////////////////////////////////////////

static void releaseInstances(
    anari::Device d, const std::vector<anari::Instance> &instances)
{
  for (auto i : instances)
    anari::release(d, i);
}

// RenderIndexAllLayers definitions ///////////////////////////////////////////

RenderIndexAllLayers::RenderIndexAllLayers(Context &ctx, anari::Device d)
    : RenderIndex(ctx, d)
{}

RenderIndexAllLayers::~RenderIndexAllLayers()
{
  releaseAllInstances();
}

void RenderIndexAllLayers::setFilterFunction(RenderIndexFilterFcn f)
{
  m_filter = f;
  m_filterForceUpdate = true;
  signalObjectFilteringChanged();
}

void RenderIndexAllLayers::signalArrayUnmapped(const Array *a)
{
  RenderIndex::signalArrayUnmapped(a);
  if (a->elementType() == ANARI_FLOAT32_MAT4)
    updateWorld();
}

void RenderIndexAllLayers::signalLayerAdded(const Layer *l)
{
  syncLayerInstances(l);
  updateWorld();
}

void RenderIndexAllLayers::signalLayerUpdated(const Layer *l)
{
  syncLayerInstances(l);
  updateWorld();
}

void RenderIndexAllLayers::signalLayerRemoved(const Layer *l)
{
  releaseInstances(device(), m_instanceCache[l]);
  m_instanceCache.erase(l);
  updateWorld();
}

void RenderIndexAllLayers::signalObjectFilteringChanged()
{
  if (m_filter || m_filterForceUpdate) {
    releaseAllInstances();
    updateWorld();
    m_filterForceUpdate = false;
  }
}

void RenderIndexAllLayers::signalRemoveAllObjects()
{
  releaseAllInstances();
  RenderIndex::signalRemoveAllObjects();
}

void RenderIndexAllLayers::updateWorld()
{
  auto d = device();
  auto w = world();

  if (m_instanceCache.empty()) {
    for (auto &l : m_ctx->layers())
      syncLayerInstances(l.second.get());
  }

  std::vector<anari::Instance> instances;
  instances.reserve(2000);

  for (auto &i : m_instanceCache)
    std::copy(i.second.begin(), i.second.end(), std::back_inserter(instances));

  if (instances.empty())
    anari::unsetParameter(d, w, "instance");
  else {
    anari::setParameterArray1D(
        d, w, "instance", instances.data(), instances.size());
  }

  anari::commitParameters(d, w);
}

void RenderIndexAllLayers::syncLayerInstances(const Layer *_layer)
{
  auto d = device();

  std::vector<anari::Instance> instances;
  instances.reserve(100);

  auto *layer = const_cast<Layer *>(_layer);

  RenderToAnariObjectsVisitor visitor(
      d, m_cache, &instances, m_ctx, m_filter ? &m_filter : nullptr);
  layer->traverse(layer->root(), visitor);

  auto &cached = m_instanceCache[layer];
  releaseInstances(d, cached);
  cached = instances;
}

void RenderIndexAllLayers::releaseAllInstances()
{
  for (auto &i : m_instanceCache)
    releaseInstances(device(), i.second);
  m_instanceCache.clear();
}

} // namespace tsd::rendering
