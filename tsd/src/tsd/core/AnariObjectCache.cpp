// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/AnariObjectCache.hpp"
#include "tsd/core/Object.hpp"

namespace tsd {

AnariObjectCache::AnariObjectCache(anari::Device d) : device(d)
{
  anari::retain(device, device);
}

AnariObjectCache::~AnariObjectCache()
{
  clear();
  anari::release(device, device);
}

anari::Object AnariObjectCache::getHandle(anari::DataType type, size_t i) const
{
  anari::Object obj = nullptr;

  switch (type) {
  case ANARI_SURFACE:
    obj = *surface.at(i);
    break;
  case ANARI_GEOMETRY:
    obj = *geometry.at(i);
    break;
  case ANARI_MATERIAL:
    obj = *material.at(i);
    break;
  case ANARI_SAMPLER:
    obj = *sampler.at(i);
    break;
  case ANARI_VOLUME:
    obj = *volume.at(i);
    break;
  case ANARI_SPATIAL_FIELD:
    obj = *field.at(i);
    break;
  case ANARI_LIGHT:
    obj = *light.at(i);
    break;
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D:
    obj = *array.at(i);
    break;
  default:
    break; // no-op
  }

  return obj;
}

anari::Object AnariObjectCache::getHandle(const Object *o) const
{
  return getHandle(o->type(), o->index());
}

void AnariObjectCache::removeHandle(anari::DataType type, size_t index)
{
  auto handle = getHandle(type, index);
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

} // namespace tsd
