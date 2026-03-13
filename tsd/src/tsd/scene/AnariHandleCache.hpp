// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ObjectPool.hpp"
#include "tsd/core/Token.hpp"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::scene {

struct Array;
struct Object;
struct Scene;

struct AnariHandleCache
{
  AnariHandleCache(Scene &scene, tsd::core::Token deviceName, anari::Device d);
  ~AnariHandleCache();
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

  tsd::core::ObjectPool<anari::Surface> surface;
  tsd::core::ObjectPool<anari::Geometry> geometry;
  tsd::core::ObjectPool<anari::Material> material;
  tsd::core::ObjectPool<anari::Sampler> sampler;
  tsd::core::ObjectPool<anari::Volume> volume;
  tsd::core::ObjectPool<anari::SpatialField> field;
  tsd::core::ObjectPool<anari::Light> light;
  tsd::core::ObjectPool<anari::Array> array;
  tsd::core::ObjectPool<anari::Renderer> renderer;
  tsd::core::ObjectPool<anari::Camera> camera;

  anari::Device device{nullptr};
  tsd::core::Token deviceName;

 private:
  void replaceHandle(anari::Object o, anari::DataType type, size_t i);
  anari::Object readHandle(anari::DataType type, size_t i) const;

  Scene *m_scene{nullptr};
  bool m_supportsCUDA{false};
};

} // namespace tsd::scene
