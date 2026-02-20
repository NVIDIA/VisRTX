// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderIndex.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <algorithm>

namespace tsd::rendering {

RenderIndex::RenderIndex(
    Scene &scene, tsd::core::Token deviceName, anari::Device d)
    : m_cache(scene, deviceName, d), m_ctx(&scene)
{
  anari::retain(d, d);
  m_world = anari::newObject<anari::World>(d);
}

RenderIndex::~RenderIndex()
{
  auto d = device();
  anari::release(d, m_world); // release before AnariObjectCache
  anari::release(d, d);
}

anari::Device RenderIndex::device() const
{
  return m_cache.device;
}

anari::World RenderIndex::world() const
{
  return m_world;
}

anari::Renderer RenderIndex::renderer(size_t i)
{
  return (anari::Renderer)m_cache.getHandle(ANARI_RENDERER, i, true);
}

void RenderIndex::logCacheInfo() const
{
  logStatus("RENDER INDEX:");
  logStatus("      device: %p", device());
  logStatus("       world: %p", world());
  logStatus("    surfaces: %zu", m_cache.surface.size());
  logStatus("  geometries: %zu", m_cache.geometry.size());
  logStatus("   materials: %zu", m_cache.material.size());
  logStatus("    samplers: %zu", m_cache.sampler.size());
  logStatus("     volumes: %zu", m_cache.volume.size());
  logStatus("      fields: %zu", m_cache.field.size());
  logStatus("      lights: %zu", m_cache.light.size());
  logStatus("      arrays: %zu", m_cache.array.size());
  logStatus("   renderers: %zu", m_cache.renderer.size());
}

void RenderIndex::populate(bool setAsUpdateDelegate)
{
  m_cache.clear();

  auto createANARICacheObjects = [&](const auto &objArray, auto &handleArray) {
    foreach_item_const(
        objArray, [&](auto *obj) { handleArray.insert(nullptr); });
    handleArray.sync_slots(objArray);
  };

  const auto &db = m_ctx->objectDB();
  createANARICacheObjects(db.array, m_cache.array);
  createANARICacheObjects(db.sampler, m_cache.sampler);
  createANARICacheObjects(db.material, m_cache.material);
  createANARICacheObjects(db.geometry, m_cache.geometry);
  createANARICacheObjects(db.surface, m_cache.surface);
  createANARICacheObjects(db.field, m_cache.field);
  createANARICacheObjects(db.volume, m_cache.volume);
  createANARICacheObjects(db.light, m_cache.light);
  createANARICacheObjects(db.renderer, m_cache.renderer);

  if (setAsUpdateDelegate)
    m_ctx->setUpdateDelegate(this);

  updateWorld();
}

void RenderIndex::setFilterFunction(RenderIndexFilterFcn f)
{
  // no-op
}

void RenderIndex::setExternalInstances(
    const anari::Instance *instances, size_t count)
{
  m_externalInstances.resize(count);
  std::copy(instances, instances + count, m_externalInstances.data());
  updateWorld();
}

void RenderIndex::signalObjectAdded(const Object *obj)
{
  if (!obj)
    return;
  m_cache.insertEmptyHandle(obj->type());
}

void RenderIndex::signalParameterUpdated(const Object *o, const Parameter *p)
{
  if (anari::Object obj = m_cache.getHandle(o, false); obj) {
    auto d = device();
    o->updateANARIParameter(d, obj, *p, p->name().c_str(), &m_cache);
    anari::commitParameters(d, obj);
  }
}

void RenderIndex::signalParameterRemoved(const Object *o, const Parameter *p)
{
  if (anari::Object obj = m_cache.getHandle(o, false); obj) {
    auto d = device();
    anari::unsetParameter(d, obj, p->name().c_str());
    anari::commitParameters(d, obj);
  }
}

void RenderIndex::signalArrayMapped(const Array *a)
{
  if (anari::isObject(a->elementType()))
    return;
  if (anari::Object obj = m_cache.getHandle(a, false); obj != nullptr)
    anariMapArray(device(), (anari::Array)obj);
}

void RenderIndex::signalArrayUnmapped(const Array *a)
{
  if (anari::isObject(a->elementType()))
    m_cache.updateObjectArrayData(a);
  else if (auto arr = (anari::Array)m_cache.getHandle(a, false); arr != nullptr)
    anariUnmapArray(device(), (anari::Array)arr);
}

void RenderIndex::signalLayerAdded(const Layer *)
{
  // no-op
}

void RenderIndex::signalLayerUpdated(const Layer *)
{
  // no-op
}

void RenderIndex::signalLayerRemoved(const Layer *)
{
  // no-op
}

void RenderIndex::signalActiveLayersChanged()
{
  // no-op
}

void RenderIndex::signalObjectFilteringChanged()
{
  // no-op
}

void RenderIndex::signalObjectRemoved(const Object *o)
{
  m_cache.removeHandle(o);
  updateWorld();
}

void RenderIndex::signalRemoveAllObjects()
{
  auto d = device();
  auto w = world();
  anari::unsetAllParameters(d, w);
  anari::commitParameters(d, w);
  m_cache.clear();
}

void RenderIndex::signalInvalidateCachedObjects()
{
  signalRemoveAllObjects();
  populate(false); // always 'false' as this may already be the delegate
  updateWorld();
}

void RenderIndex::signalAnimationTimeChanged(float)
{
  // no-op
}

} // namespace tsd::rendering
