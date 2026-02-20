// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ObjectPool.hpp"
#include "tsd/core/Token.hpp"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::core {

struct Array;
struct Object;
struct Scene;

struct AnariObjectCache
{
  AnariObjectCache(Scene &scene, tsd::core::Token deviceName, anari::Device d);
  ~AnariObjectCache();
  anari::Object getHandle(
      anari::DataType type, size_t index, bool createIfNotPresent);
  anari::Object getHandle(const Object *o, bool createIfNotPresent);
  void insertEmptyHandle(anari::DataType type);
  void releaseHandle(anari::DataType type, size_t index);
  void releaseHandle(const Object *o);
  void removeHandle(anari::DataType type, size_t index);
  void removeHandle(const Object *o);
  void clear();
  bool supportsCUDA() const;
  void updateObjectArrayData(const Array *a); // for arrays-of-arrays

  ObjectPool<anari::Surface> surface;
  ObjectPool<anari::Geometry> geometry;
  ObjectPool<anari::Material> material;
  ObjectPool<anari::Sampler> sampler;
  ObjectPool<anari::Volume> volume;
  ObjectPool<anari::SpatialField> field;
  ObjectPool<anari::Light> light;
  ObjectPool<anari::Array> array;
  ObjectPool<anari::Renderer> renderer;

  anari::Device device{nullptr};
  tsd::core::Token deviceName;

 private:
  void replaceHandle(anari::Object o, anari::DataType type, size_t i);
  anari::Object readHandle(anari::DataType type, size_t i) const;

  Scene *m_scene{nullptr};
  bool m_supportsCUDA{false};
};

} // namespace tsd::core
