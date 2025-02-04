// Copyright 2024-2025 NVIDIA Corporation
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
  anari::Object getHandle(anari::DataType type, size_t index) const;
  anari::Object getHandle(const Object *o) const;
  void removeHandle(anari::DataType type, size_t index);
  void removeHandle(const Object *o);
  void clear();
  bool supportsCUDA() const;

  IndexedVector<anari::Surface> surface;
  IndexedVector<anari::Geometry> geometry;
  IndexedVector<anari::Material> material;
  IndexedVector<anari::Sampler> sampler;
  IndexedVector<anari::Volume> volume;
  IndexedVector<anari::SpatialField> field;
  IndexedVector<anari::Light> light;
  IndexedVector<anari::Array> array;

  anari::Device device{nullptr};

private:
  bool m_supportsCUDA{false};
};

} // namespace tsd