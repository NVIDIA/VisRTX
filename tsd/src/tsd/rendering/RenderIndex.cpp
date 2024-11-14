// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderIndex.hpp"
#include "tsd/core/Logging.hpp"

namespace tsd {

RenderIndex::RenderIndex(anari::Device d) : m_cache(d)
{
  anari::retain(d, d);
  m_world = anari::newObject<anari::World>(d);
}

RenderIndex::~RenderIndex()
{
#if 0 // NOTE(jda) - should be ok, but causes OSPRay to crash on exit
  signalRemoveAllObjects();
#endif
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
}

void RenderIndex::populate(Context &ctx, bool setAsUpdateDelegate)
{
  m_cache.clear();

  auto d = device();
  auto w = world();
  m_ctx = &ctx;

  const auto &db = ctx.objectDB();

  // Setup individual leaf object handles //

  auto setupANARICacheArray = [&](const auto &objArray, auto &handleArray) {
    foreach_item_const(objArray, [&](auto *obj) {
      if (!obj) {
        handleArray.insert({});
        return;
      }

      using handle_t = typename std::remove_reference<
          decltype(handleArray)>::type::element_t;
      auto o = (handle_t)obj->makeANARIObject(d);
      obj->updateAllANARIParameters(d, o, &m_cache);
      if (obj->type() == ANARI_SURFACE)
        anari::setParameter(d, o, "id", uint32_t(obj->index()));
      anari::commitParameters(d, o);
      handleArray.insert(std::move(o));
    });

    handleArray.sync_slots(objArray);
  };

  // NOTE: These needs to go from bottom to top of the ANARI hierarchy!
  setupANARICacheArray(db.array, m_cache.array);
  setupANARICacheArray(db.sampler, m_cache.sampler);
  setupANARICacheArray(db.material, m_cache.material);
  setupANARICacheArray(db.geometry, m_cache.geometry);
  setupANARICacheArray(db.surface, m_cache.surface);
  setupANARICacheArray(db.field, m_cache.field);
  setupANARICacheArray(db.volume, m_cache.volume);
  setupANARICacheArray(db.light, m_cache.light);

  // NOTE: ensure that object arrays are properly populated
  foreach_item_const(db.array, [&](auto *a) { updateObjectArrayData(a); });

  if (setAsUpdateDelegate)
    ctx.setUpdateDelegate(this);

  updateWorld();
}

void RenderIndex::setFilterFunction(RenderIndexFilterFcn f)
{
  // no-op
}

void RenderIndex::signalObjectAdded(const Object *obj)
{
  if (!obj)
    return;

  auto d = device();
  auto o = obj->makeANARIObject(d);
  obj->updateAllANARIParameters(d, o, &m_cache);

  switch (obj->type()) {
  case ANARI_SURFACE: {
    auto s = m_cache.surface.insert((anari::Surface)o);
    anari::setParameter(d, o, "id", uint32_t(obj->index()));
  } break;
  case ANARI_GEOMETRY:
    m_cache.geometry.insert((anari::Geometry)o);
    break;
  case ANARI_MATERIAL:
    m_cache.material.insert((anari::Material)o);
    break;
  case ANARI_SAMPLER:
    m_cache.sampler.insert((anari::Sampler)o);
    break;
  case ANARI_VOLUME:
    m_cache.volume.insert((anari::Volume)o);
    break;
  case ANARI_SPATIAL_FIELD:
    m_cache.field.insert((anari::SpatialField)o);
    break;
  case ANARI_LIGHT:
    m_cache.light.insert((anari::Light)o);
    break;
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D:
    m_cache.array.insert((anari::Array)o);
    break;
  default:
    break; // no-op
  }

  anari::commitParameters(d, o);
}

void RenderIndex::signalParameterUpdated(const Object *o, const Parameter *p)
{
  if (anari::Object obj = m_cache.getHandle(o); obj) {
    auto d = device();
    o->updateANARIParameter(d, obj, *p, p->name().c_str(), &m_cache);
    anari::commitParameters(d, obj);
  }
}

void RenderIndex::signalParameterRemoved(const Object *o, const Parameter *p)
{
  if (anari::Object obj = m_cache.getHandle(o); obj) {
    auto d = device();
    anari::unsetParameter(d, obj, p->name().c_str());
    anari::commitParameters(d, obj);
  }
}

void RenderIndex::signalArrayMapped(const Array *a)
{
  if (anari::isObject(a->elementType()))
    return;

  if (anari::Object obj = m_cache.getHandle(a); obj != nullptr)
    anariMapArray(device(), (anari::Array)obj);
}

void RenderIndex::signalArrayUnmapped(const Array *a)
{
  if (anari::isObject(a->elementType()))
    updateObjectArrayData(a);
  else if (auto arr = (anari::Array)m_cache.getHandle(a); arr != nullptr)
    anariUnmapArray(device(), (anari::Array)arr);
}

void RenderIndex::signalInstanceStructureChanged()
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

void RenderIndex::updateObjectArrayData(const Array *a) const
{
  auto elementType = a->elementType();
  if (!a || !anari::isObject(elementType))
    return;

  if (auto arr = (anari::Array)m_cache.getHandle(a); arr != nullptr) {
    auto *src = a->dataAs<size_t>();
    auto *dst = (anari::Object *)anariMapArray(device(), arr);
    std::transform(src, src + a->size(), dst, [&](size_t idx) {
      auto handle = m_cache.getHandle(elementType, idx);
      if (handle == nullptr)
        logWarning("[RenderIndex] object array encountered null handle");
      return handle;
    });
    anariUnmapArray(device(), arr);
  }
}

} // namespace tsd
