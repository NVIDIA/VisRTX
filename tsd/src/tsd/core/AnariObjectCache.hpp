// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/containers/IndexedVector.hpp"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd {

struct Object;

struct AnariObjectCache
{
  AnariObjectCache(anari::Device d);
  ~AnariObjectCache();
  anari::Object getHandle(ANARIDataType type, size_t index) const;
  anari::Object getHandle(const Object *o) const;
  void removeHandle(ANARIDataType type, size_t index);
  void removeHandle(const Object *o);
  void clear();

  IndexedVector<anari::Surface> surface;
  IndexedVector<anari::Geometry> geometry;
  IndexedVector<anari::Material> material;
  IndexedVector<anari::Sampler> sampler;
  IndexedVector<anari::Volume> volume;
  IndexedVector<anari::SpatialField> field;
  IndexedVector<anari::Light> light;
  IndexedVector<anari::Array> array;

  anari::Device device{nullptr};
};

} // namespace tsd