// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/AnariObjectCache.hpp"
#include "tsd/core/Object.hpp"
#include "tsd/core/Logging.hpp"

namespace tsd {

// Helper functions ///////////////////////////////////////////////////////////

static bool supportsCUDAArrays(anari::Device d)
{
  bool supportsCUDA = false;
  auto list = (const char *const *)anariGetObjectInfo(
    d,
    ANARI_DEVICE,
    "default",
    "extension",
    ANARI_STRING_LIST
  );

  for(const char *const *i = list; *i != nullptr; ++i) {
    if (std::string(*i) == "ANARI_VISRTX_ARRAY_CUDA") {
      supportsCUDA = true;
      break;
    }
  }

  return supportsCUDA;
}

// AnariObjectCache definitions ///////////////////////////////////////////////

AnariObjectCache::AnariObjectCache(anari::Device d) : device(d)
{
  anari::retain(device, device);
  m_supportsCUDA = supportsCUDAArrays(d);
  tsd::logStatus("[ANARI object cache] device %s CUDA arrays",
    m_supportsCUDA ? "supports" : "does NOT support");
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

bool AnariObjectCache::supportsCUDA() const
{
  return m_supportsCUDA;
}

} // namespace tsd
