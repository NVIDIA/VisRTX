// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/TypeMacros.hpp"
// tsd_scene
#include "Layer.hpp"
// std
#include <memory>
#include <vector>

namespace tsd::scene {

struct Array;
struct Object;
struct Parameter;

/*
 * Abstract observer interface that receives all mutating signals produced by
 * a Scene: object creation/removal, parameter changes, and layer edits.
 * Subclass to drive downstream systems (e.g. renderers).
 *
 * Example:
 *   struct MyDelegate : BaseUpdateDelegate {
 *     void signalParameterUpdated(const Object *o, const Parameter *p) override
 *       { scheduleUpload(o, p); }
 *     // ... implement remaining pure virtuals ...
 *   };
 *   scene.setUpdateDelegate(&myDelegate);
 */
struct BaseUpdateDelegate
{
  BaseUpdateDelegate() = default;
  virtual ~BaseUpdateDelegate() = default;

  virtual void signalObjectAdded(const Object *o) = 0;
  virtual void signalParameterUpdated(const Object *o, const Parameter *p) = 0;
  virtual void signalParameterRemoved(const Object *o, const Parameter *p) = 0;
  virtual void signalParameterBatchUpdated(
      const Object *o, const std::vector<const Parameter *> &ps) = 0;
  virtual void signalArrayMapped(const Array *a) = 0;
  virtual void signalArrayUnmapped(const Array *a) = 0;
  virtual void signalObjectParameterUseCountZero(const Object *obj) = 0;
  virtual void signalObjectLayerUseCountZero(const Object *obj) = 0;
  virtual void signalObjectRemoved(const Object *o) = 0;
  virtual void signalRemoveAllObjects() = 0;
  virtual void signalLayerAdded(const Layer *l) = 0;
  virtual void signalLayerStructureUpdated(const Layer *l) = 0;
  virtual void signalLayerTransformUpdated(const Layer *l) = 0;
  virtual void signalLayerRemoved(const Layer *l) = 0;
  virtual void signalActiveLayersChanged() = 0;
  virtual void signalObjectFilteringChanged() = 0;
  virtual void signalInvalidateCachedObjects() = 0;

  TSD_NOT_COPYABLE(BaseUpdateDelegate)
  TSD_DEFAULT_MOVEABLE(BaseUpdateDelegate)
};

/*
 * Concrete BaseUpdateDelegate that silently discards every signal; useful as a
 * placeholder or base for delegates that only need to handle a subset of
 * events.
 *
 * Example:
 *   EmptyUpdateDelegate noop;
 *   scene.setUpdateDelegate(&noop);
 */
struct EmptyUpdateDelegate : public BaseUpdateDelegate
{
  EmptyUpdateDelegate() = default;
  virtual ~EmptyUpdateDelegate() override = default;

  void signalObjectAdded(const Object *) override {}
  void signalParameterUpdated(const Object *, const Parameter *) override {}
  void signalParameterRemoved(const Object *, const Parameter *) override {}
  void signalParameterBatchUpdated(
      const Object *, const std::vector<const Parameter *> &) override
  {}
  void signalArrayMapped(const Array *) override {}
  void signalArrayUnmapped(const Array *) override {}
  void signalObjectParameterUseCountZero(const Object *obj) override {};
  void signalObjectLayerUseCountZero(const Object *obj) override {};
  void signalObjectRemoved(const Object *) override {}
  void signalRemoveAllObjects() override {}
  void signalLayerAdded(const Layer *) override {}
  void signalLayerStructureUpdated(const Layer *) override {}
  void signalLayerTransformUpdated(const Layer *) override {}
  void signalLayerRemoved(const Layer *) override {}
  void signalActiveLayersChanged() override {}
  void signalObjectFilteringChanged() override {}
  void signalInvalidateCachedObjects() override {}
};

/*
 * BaseUpdateDelegate that owns a list of child delegates and fans every signal
 * out to each of them; enables multiple independent systems to observe a Scene.
 *
 * Example:
 *   MultiUpdateDelegate multi;
 *   multi.emplace<RenderDelegate>();
 *   multi.emplace<NetworkDelegate>();
 *   scene.setUpdateDelegate(&multi);
 */
struct MultiUpdateDelegate : public BaseUpdateDelegate
{
  MultiUpdateDelegate() = default;
  ~MultiUpdateDelegate() override = default;

  template <typename T, typename... Args>
  T *emplace(Args &&...args);
  size_t size() const;
  void clear();
  void erase(const BaseUpdateDelegate *d);

  const BaseUpdateDelegate *get(size_t i) const;
  BaseUpdateDelegate *get(size_t i);

  const BaseUpdateDelegate *operator[](size_t i) const;
  BaseUpdateDelegate *operator[](size_t i);

  void signalObjectAdded(const Object *o) override;
  void signalParameterUpdated(const Object *o, const Parameter *p) override;
  void signalParameterRemoved(const Object *o, const Parameter *p) override;
  void signalParameterBatchUpdated(
      const Object *o, const std::vector<const Parameter *> &ps) override;
  void signalArrayMapped(const Array *a) override;
  void signalArrayUnmapped(const Array *a) override;
  void signalObjectParameterUseCountZero(const Object *obj) override;
  void signalObjectLayerUseCountZero(const Object *obj) override;
  void signalObjectRemoved(const Object *o) override;
  void signalRemoveAllObjects() override;
  void signalLayerAdded(const Layer *) override;
  void signalLayerStructureUpdated(const Layer *) override;
  void signalLayerTransformUpdated(const Layer *) override;
  void signalLayerRemoved(const Layer *) override;
  void signalActiveLayersChanged() override;
  void signalObjectFilteringChanged() override;
  void signalInvalidateCachedObjects() override;

 private:
  std::vector<std::unique_ptr<BaseUpdateDelegate>> m_delegates;
};

// Inline definitions /////////////////////////////////////////////////////////

template <typename T, typename... Args>
inline T *MultiUpdateDelegate::emplace(Args &&...args)
{
  m_delegates.push_back(std::make_unique<T>(std::forward<Args>(args)...));
  return (T *)m_delegates.back().get();
}

} // namespace tsd::scene