// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/AnariObjectCache.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/scene/Scene.hpp"

namespace tsd::core {

// Helper functions ///////////////////////////////////////////////////////////

static bool supportsCUDAArrays(anari::Device d)
{
  bool supportsCUDA = false;
  auto list = (const char *const *)anariGetObjectInfo(
      d, ANARI_DEVICE, "default", "extension", ANARI_STRING_LIST);

  for (const char *const *i = list; *i != nullptr; ++i) {
    if (std::string(*i) == "ANARI_NV_ARRAY_CUDA") {
      supportsCUDA = true;
      break;
    }
  }

  return supportsCUDA;
}

// AnariObjectCache definitions ///////////////////////////////////////////////

AnariObjectCache::AnariObjectCache(Scene &scene, anari::Device d)
    : device(d), m_ctx(&scene)
{
  anari::retain(device, device);
  m_supportsCUDA = supportsCUDAArrays(d);
  tsd::core::logStatus("[ANARI object cache] device %s CUDA arrays",
      m_supportsCUDA ? "supports" : "does NOT support");
}

AnariObjectCache::~AnariObjectCache()
{
  clear();
  anari::release(device, device);
}

anari::Object AnariObjectCache::getHandle(
    anari::DataType type, size_t i, bool createIfNotPresent)
{
  return getHandle(m_ctx->getObject(type, i), createIfNotPresent);
}

anari::Object AnariObjectCache::getHandle(
    const Object *obj, bool createIfNotPresent)
{
  if (!obj)
    return nullptr;
  auto type = obj->type();
  auto idx = obj->index();
  auto o = readHandle(type, idx);
  if (!o && createIfNotPresent) {
    auto d = device;
    o = obj->makeANARIObject(d);
    obj->updateAllANARIParameters(d, o, this);
    if (type == ANARI_SURFACE)
      anari::setParameter(d, o, "id", uint32_t(idx));
    else if (type == ANARI_VOLUME)
      anari::setParameter(d, o, "id", uint32_t(idx) | 0x80000000u);
    anari::commitParameters(d, o);
    this->replaceHandle(o, type, idx);
    if (anari::isArray(type))
      updateObjectArrayData((const Array *)obj);
  }
  return o;
}

void AnariObjectCache::insertEmptyHandle(anari::DataType type)
{
  switch (type) {
  case ANARI_SURFACE:
    surface.insert(nullptr);
    break;
  case ANARI_GEOMETRY:
    geometry.insert(nullptr);
    break;
  case ANARI_MATERIAL:
    material.insert(nullptr);
    break;
  case ANARI_SAMPLER:
    sampler.insert(nullptr);
    break;
  case ANARI_VOLUME:
    volume.insert(nullptr);
    break;
  case ANARI_SPATIAL_FIELD:
    field.insert(nullptr);
    break;
  case ANARI_LIGHT:
    light.insert(nullptr);
    break;
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D:
    array.insert(nullptr);
    break;
  default:
    break; // no-op
  }
}

void AnariObjectCache::releaseHandle(anari::DataType type, size_t index)
{
  removeHandle(type, index);
  insertEmptyHandle(type);
}

void AnariObjectCache::releaseHandle(const Object *o)
{
  if (o)
    return releaseHandle(o->type(), o->index());
}

void AnariObjectCache::removeHandle(anari::DataType type, size_t index)
{
  auto handle = getHandle(type, index, false);
  anari::release(device, handle);

  switch (type) {
  case ANARI_SURFACE:
    surface.erase(index);
    break;
  case ANARI_GEOMETRY:
    geometry.erase(index);
    break;
  case ANARI_MATERIAL:
    material.erase(index);
    break;
  case ANARI_SAMPLER:
    sampler.erase(index);
    break;
  case ANARI_VOLUME:
    volume.erase(index);
    break;
  case ANARI_SPATIAL_FIELD:
    field.erase(index);
    break;
  case ANARI_LIGHT:
    light.erase(index);
    break;
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D:
    array.erase(index);
    break;
  default:
    break; // no-op
  }
}

void AnariObjectCache::removeHandle(const Object *o)
{
  return removeHandle(o->type(), o->index());
}

void AnariObjectCache::clear()
{
  auto releaseAllHandles = [&](auto &iv) {
    for (int i = 0; i < iv.capacity(); i++) {
      anari::release(device, iv[i]);
      iv[i] = nullptr;
    }
    iv.clear();
  };

  releaseAllHandles(volume);
  releaseAllHandles(field);
  releaseAllHandles(surface);
  releaseAllHandles(geometry);
  releaseAllHandles(material);
  releaseAllHandles(sampler);
  releaseAllHandles(light);
  releaseAllHandles(array); // this needs to be last!
}

bool AnariObjectCache::supportsCUDA() const
{
  return m_supportsCUDA;
}

void AnariObjectCache::updateObjectArrayData(const Array *a)
{
  auto elementType = a->elementType();
  if (!a || !anari::isObject(elementType) || a->isEmpty())
    return;

  if (auto arr = (anari::Array)this->getHandle(a, false); arr != nullptr) {
    auto *src = (const size_t *)a->data();
    auto *dst = (anari::Object *)anariMapArray(device, arr);
    std::transform(src, src + a->size(), dst, [&](size_t idx) {
      auto *obj = m_ctx->getObject(elementType, idx);
      auto handle = this->getHandle(obj, true);
      if (handle == nullptr)
        logWarning("[RenderIndex] object array encountered null handle");
      return handle;
    });
    anariUnmapArray(device, arr);
  }
}

void AnariObjectCache::replaceHandle(
    anari::Object o, anari::DataType type, size_t i)
{
  switch (type) {
  case ANARI_SURFACE:
    surface[i] = (anari::Surface)o;
    break;
  case ANARI_GEOMETRY:
    geometry[i] = (anari::Geometry)o;
    break;
  case ANARI_MATERIAL:
    material[i] = (anari::Material)o;
    break;
  case ANARI_SAMPLER:
    sampler[i] = (anari::Sampler)o;
    break;
  case ANARI_VOLUME:
    volume[i] = (anari::Volume)o;
    break;
  case ANARI_SPATIAL_FIELD:
    field[i] = (anari::SpatialField)o;
    break;
  case ANARI_LIGHT:
    light[i] = (anari::Light)o;
    break;
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D:
    array[i] = (anari::Array)o;
    break;
  default:
    break; // no-op
  }
}

anari::Object AnariObjectCache::readHandle(anari::DataType type, size_t i) const
{
  anari::Object obj = nullptr;

  switch (type) {
  case ANARI_SURFACE:
    obj = surface.at(i).value_or(nullptr);
    break;
  case ANARI_GEOMETRY:
    obj = geometry.at(i).value_or(nullptr);
    break;
  case ANARI_MATERIAL:
    obj = material.at(i).value_or(nullptr);
    break;
  case ANARI_SAMPLER:
    obj = sampler.at(i).value_or(nullptr);
    break;
  case ANARI_VOLUME:
    obj = volume.at(i).value_or(nullptr);
    break;
  case ANARI_SPATIAL_FIELD:
    obj = field.at(i).value_or(nullptr);
    break;
  case ANARI_LIGHT:
    obj = light.at(i).value_or(nullptr);
    break;
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D:
    obj = array.at(i).value_or(nullptr);
    break;
  default:
    break; // no-op
  }

  return obj;
}

} // namespace tsd::core
