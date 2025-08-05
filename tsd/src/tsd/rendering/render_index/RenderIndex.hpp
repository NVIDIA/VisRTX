// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/AnariObjectCache.hpp"
#include "tsd/core/scene/Context.hpp"
#include "tsd/core/scene/UpdateDelegate.hpp"
// std
#include <functional>

namespace tsd::rendering {

using namespace tsd::core;

struct RenderToAnariObjectsVisitor;

using RenderIndexFilterFcn = std::function<bool(const Object *)>;

struct RenderIndex : public BaseUpdateDelegate
{
  RenderIndex(Context &ctx, anari::Device d);
  virtual ~RenderIndex();

  anari::Device device() const;
  anari::World world() const;

  void logCacheInfo() const;

  void populate(bool setAsUpdateDelegate = true);

  virtual void setFilterFunction(RenderIndexFilterFcn f);

  // Handle UpdateDelegate signals //

  void signalObjectAdded(const Object *o) override;
  void signalParameterUpdated(const Object *o, const Parameter *p) override;
  void signalParameterRemoved(const Object *o, const Parameter *p) override;
  void signalArrayMapped(const Array *a) override;
  void signalArrayUnmapped(const Array *a) override;
  void signalLayerAdded(const Layer *l) override;
  void signalLayerUpdated(const Layer *l) override;
  void signalLayerRemoved(const Layer *l) override;
  void signalObjectFilteringChanged() override;
  void signalObjectRemoved(const Object *o) override;
  void signalRemoveAllObjects() override;
  void signalInvalidateCachedObjects() override;

 protected:
  virtual void updateWorld() = 0;

  Context *m_ctx{nullptr};
  AnariObjectCache m_cache;

  anari::World m_world{nullptr};

 private:
  friend struct RenderToAnariObjectsVisitor;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline void setIndexedArrayObjectsAsAnariObjectArray(
    anari::Device d, anari::Object o, const char *p, const IndexedVector<T> &iv)
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
