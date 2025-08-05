// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/IndexedVector.hpp"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::core {

struct Array;
struct Object;
struct Context;

struct AnariObjectCache
{
  AnariObjectCache(Context &ctx, anari::Device d);
  ~AnariObjectCache();
  anari::Object getHandle(
      anari::DataType type, size_t index, bool createIfNotPresent);
  anari::Object getHandle(const Object *o, bool createIfNotPresent);
  void insertEmptyHandle(anari::DataType type);
  void removeHandle(anari::DataType type, size_t index);
  void removeHandle(const Object *o);
  void clear();
  bool supportsCUDA() const;
  void updateObjectArrayData(const Array *a); // for arrays-of-arrays

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
  void replaceHandle(anari::Object o, anari::DataType type, size_t i);
  anari::Object readHandle(anari::DataType type, size_t i) const;

  Context *m_ctx{nullptr};
  bool m_supportsCUDA{false};
};

} // namespace tsd::core
