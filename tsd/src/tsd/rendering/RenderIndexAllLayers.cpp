// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/rendering/RenderIndexAllLayers.hpp"
// std
#include <algorithm>
#include <iterator>
#include <stack>

namespace tsd {

// Helper functions ///////////////////////////////////////////////////////////

static void releaseInstances(
    anari::Device d, const std::vector<anari::Instance> &instances)
{
  for (auto i : instances)
    anari::release(d, i);
}

// Helper types ///////////////////////////////////////////////////////////////

struct RenderToAnariObjectsVisitor : public LayerVisitor
{
  RenderToAnariObjectsVisitor(anari::Device d,
      AnariObjectCache &oc,
      std::vector<anari::Instance> *instances,
      Context *ctx,
      RenderIndexFilterFcn *f = nullptr);
  ~RenderToAnariObjectsVisitor();

  bool preChildren(LayerNode &n, int level) override;
  void postChildren(LayerNode &n, int level) override;

 private:
  bool isIncludedAfterFiltering(const LayerNode &n) const;
  void createInstanceFromTop();

  struct GroupedObjects
  {
    std::vector<anari::Surface> surfaces;
    std::vector<anari::Volume> volumes;
    std::vector<anari::Light> lights;
  };

  anari::Device m_device{nullptr};
  AnariObjectCache *m_cache{nullptr};
  std::vector<anari::Instance> *m_instances{nullptr};
  std::stack<tsd::mat4> m_xfms;
  std::stack<GroupedObjects> m_objects;
  const Array *m_xfmArray{nullptr};
  Context *m_ctx{nullptr};
  RenderIndexFilterFcn *m_filter{nullptr};
};

// Inlined definitions //

inline RenderToAnariObjectsVisitor::RenderToAnariObjectsVisitor(anari::Device d,
    AnariObjectCache &oc,
    std::vector<anari::Instance> *instances,
    Context *ctx,
    RenderIndexFilterFcn *f)
    : m_device(d), m_cache(&oc), m_instances(instances), m_ctx(ctx), m_filter(f)
{
  anari::retain(d, d);
  m_xfms.emplace(tsd::math::identity);
  m_objects.emplace();
}

inline RenderToAnariObjectsVisitor::~RenderToAnariObjectsVisitor()
{
  anari::release(m_device, m_device);
}

inline bool RenderToAnariObjectsVisitor::preChildren(LayerNode &n, int level)
{
  if (!n->enabled)
    return false;

  auto &current = m_objects.top();

  const bool included = isIncludedAfterFiltering(n);

  switch (n->value.type()) {
  case ANARI_SURFACE: {
    size_t i = n->value.getAsObjectIndex();
    if (auto h = m_cache->surface[i]; h != nullptr && included)
      current.surfaces.push_back(h);
  } break;
  case ANARI_VOLUME: {
    size_t i = n->value.getAsObjectIndex();
    if (auto h = m_cache->volume[i]; h != nullptr && included)
      current.volumes.push_back(h);
  } break;
  case ANARI_LIGHT: {
    size_t i = n->value.getAsObjectIndex();
    if (auto h = m_cache->light[i]; h != nullptr)
      current.lights.push_back(h);
  } break;
  case ANARI_FLOAT32_MAT4:
    m_xfms.push(tsd::math::mul(m_xfms.top(), n->value.get<tsd::mat4>()));
    m_objects.emplace();
    break;
  case ANARI_ARRAY1D: {
    auto *a = (const Array *)m_ctx->getObject(n->value);
    if (a && a->elementType() == ANARI_FLOAT32_MAT4) {
      m_objects.emplace();
      m_xfmArray = a;
    }
  }
  default:
    break;
  }

  return true;
}

inline void RenderToAnariObjectsVisitor::postChildren(LayerNode &n, int level)
{
  if (!n->enabled)
    return;

  bool consumeXfmArray = false;
  switch (n->value.type()) {
  case ANARI_ARRAY1D: {
    auto *a = (Array *)m_ctx->getObject(n->value);
    if (!a || a->elementType() != ANARI_FLOAT32_MAT4)
      break;
    consumeXfmArray = true;
  }
  // intentionally fallthrough...
  case ANARI_FLOAT32_MAT4:
    createInstanceFromTop();

    if (!consumeXfmArray)
      m_xfms.pop();
    else {
      //
      // NOTE(jda) - custom parameters here is awkward, should be generalized...
      //
      //   TODO: Put setting TSD object parameters on an ANARI handle in a
      //         common spot for here + base Object updates physically the same.
      //
      anari::Instance inst = m_instances->back();
      for (auto &p : n->customParameters) {
        if (!p.second.holdsObject())
          continue;
        auto objType = p.second.type();
        auto objHandle =
            m_cache->getHandle(objType, p.second.getAsObjectIndex());
        anari::setParameter(
            m_device, inst, p.first.c_str(), objType, &objHandle);
      }
      anari::commitParameters(m_device, inst);
    }

    m_objects.pop();
    break;
  default:
    // no-op
    break;
  }
}

bool RenderToAnariObjectsVisitor::isIncludedAfterFiltering(
    const LayerNode &n) const
{
  if (!m_filter)
    return true;

  auto type = n->value.type();
  if (!anari::isObject(type))
    return false;

  size_t i = n->value.getAsObjectIndex();
  return (*m_filter)(m_ctx->getObject(type, i));
}

inline void RenderToAnariObjectsVisitor::createInstanceFromTop()
{
  auto &current = m_objects.top();
  if (current.surfaces.empty() && current.volumes.empty()
      && current.lights.empty())
    return;

  auto group = anari::newObject<anari::Group>(m_device);

  if (!current.surfaces.empty()) {
    anari::setParameterArray1D(m_device,
        group,
        "surface",
        current.surfaces.data(),
        current.surfaces.size());
  }

  if (!current.volumes.empty()) {
    anari::setParameterArray1D(m_device,
        group,
        "volume",
        current.volumes.data(),
        current.volumes.size());
  }

  if (!current.lights.empty()) {
    anari::setParameterArray1D(
        m_device, group, "light", current.lights.data(), current.lights.size());
  }

  current.surfaces.clear();
  current.volumes.clear();
  current.lights.clear();

  anari::commitParameters(m_device, group);

  auto instance = anari::newObject<anari::Instance>(m_device, "transform");

  const auto xfm = m_xfms.top();
  if (!m_xfmArray)
    anari::setParameter(m_device, instance, "transform", xfm);
  else {
    const auto *xfms_in = m_xfmArray->dataAs<tsd::mat4>();

    uint64_t stride = 0;
    auto *xfms_out = (tsd::mat4 *)anariMapParameterArray1D(m_device,
        instance,
        "transform",
        ANARI_FLOAT32_MAT4,
        m_xfmArray->size(),
        &stride);

    if (stride == sizeof(tsd::mat4)) {
      std::transform(xfms_in,
          xfms_in + m_xfmArray->size(),
          xfms_out,
          [&](const tsd::mat4 &m) { return tsd::math::mul(xfm, m); });
    } else {
      throw std::runtime_error("render index -- bad transform array stride");
    }

    anariUnmapParameterArray(m_device, instance, "transform");

    m_xfmArray = nullptr;
  }
  anari::setParameter(m_device, instance, "group", group);
  anari::commitParameters(m_device, instance);
  m_instances->push_back(instance);

  anari::release(m_device, group);
}

// RenderIndexAllLayers definitions ///////////////////////////////////////////

RenderIndexAllLayers::RenderIndexAllLayers(anari::Device d) : RenderIndex(d) {}

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

} // namespace tsd
