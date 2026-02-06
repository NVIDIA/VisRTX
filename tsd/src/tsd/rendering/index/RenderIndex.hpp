// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/AnariObjectCache.hpp"
#include "tsd/core/scene/Scene.hpp"
#include "tsd/core/scene/UpdateDelegate.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndexFilterFcn.hpp"

namespace tsd::rendering {

using namespace tsd::core;

struct RenderToAnariObjectsVisitor;

struct RenderIndex : public BaseUpdateDelegate
{
  RenderIndex(Scene &scene, anari::Device d);
  virtual ~RenderIndex();

  anari::Device device() const;
  anari::World world() const;

  void logCacheInfo() const;

  void populate(bool setAsUpdateDelegate = true);

  virtual void setFilterFunction(RenderIndexFilterFcn f);
  virtual bool isFlat() const = 0;

  void setExternalInstances(
      const anari::Instance *instances = nullptr, size_t count = 0);

  // Handle UpdateDelegate signals //

  void signalObjectAdded(const Object *o) override;
  void signalParameterUpdated(const Object *o, const Parameter *p) override;
  void signalParameterRemoved(const Object *o, const Parameter *p) override;
  void signalArrayMapped(const Array *a) override;
  void signalArrayUnmapped(const Array *a) override;
  void signalLayerAdded(const Layer *l) override;
  void signalLayerUpdated(const Layer *l) override;
  void signalLayerRemoved(const Layer *l) override;
  void signalActiveLayersChanged() override;
  void signalObjectFilteringChanged() override;
  void signalObjectRemoved(const Object *o) override;
  void signalRemoveAllObjects() override;
  void signalInvalidateCachedObjects() override;
  void signalAnimationTimeChanged(float time) override;

 protected:
  virtual void updateWorld() = 0;

  Scene *m_ctx{nullptr};
  AnariObjectCache m_cache;

  anari::World m_world{nullptr};
  std::vector<anari::Instance> m_externalInstances;

 private:
  friend struct RenderToAnariObjectsVisitor;
};

using MultiRenderIndex = tsd::core::MultiUpdateDelegate;

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline void setIndexedArrayObjectsAsAnariObjectArray(
    anari::Device d, anari::Object o, const char *p, const ObjectPool<T> &iv)
{
  if (iv.empty()) {
    anari::unsetParameter(d, o, p);
    return;
  }

  uint64_t stride = 0;
  auto *handles = (anari::Object *)anariMapParameterArray1D(
      d, o, p, anari::ANARITypeFor<T>::value, iv.size(), &stride);

  if (stride != sizeof(anari::Object))
    throw std::runtime_error("encountered non-dense object array stride");

  size_t i = 0, j = 0;
  for (; i < iv.capacity(); i++) {
    if (auto obj = iv[i]; !iv.slot_empty(i) && obj != nullptr)
      handles[j++] = obj;
  }

  assert(j == iv.size());

  anariUnmapParameterArray(d, o, p);
}

} // namespace tsd::rendering
