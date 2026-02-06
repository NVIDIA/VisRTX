// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Layer.hpp"
// std
#include <memory>
#include <vector>

namespace tsd::core {

struct Array;
struct Object;
struct Parameter;

/// UpdateDelegate interface containing all signals from items in a TSD context
struct BaseUpdateDelegate
{
  BaseUpdateDelegate() = default;
  virtual ~BaseUpdateDelegate() = default;

  virtual void signalObjectAdded(const Object *o) = 0;
  virtual void signalParameterUpdated(const Object *o, const Parameter *p) = 0;
  virtual void signalParameterRemoved(const Object *o, const Parameter *p) = 0;
  virtual void signalArrayMapped(const Array *a) = 0;
  virtual void signalArrayUnmapped(const Array *a) = 0;
  virtual void signalObjectParameterUseCountZero(const Object *obj) = 0;
  virtual void signalObjectLayerUseCountZero(const Object *obj) = 0;
  virtual void signalObjectRemoved(const Object *o) = 0;
  virtual void signalRemoveAllObjects() = 0;
  virtual void signalLayerAdded(const Layer *l) = 0;
  virtual void signalLayerUpdated(const Layer *l) = 0;
  virtual void signalLayerRemoved(const Layer *l) = 0;
  virtual void signalActiveLayersChanged() = 0;
  virtual void signalObjectFilteringChanged() = 0;
  virtual void signalInvalidateCachedObjects() = 0;
  virtual void signalAnimationTimeChanged(float time) = 0;

  // Not copyable or movable
  BaseUpdateDelegate(const BaseUpdateDelegate &) = delete;
  BaseUpdateDelegate &operator=(const BaseUpdateDelegate &) = delete;
  BaseUpdateDelegate(BaseUpdateDelegate &&) = default;
  BaseUpdateDelegate &operator=(BaseUpdateDelegate &&) = default;
};

/// Update delegate that implements a 'no-op' for all signals
struct EmptyUpdateDelegate : public BaseUpdateDelegate
{
  EmptyUpdateDelegate() = default;
  virtual ~EmptyUpdateDelegate() override = default;

  void signalObjectAdded(const Object *) override {}
  void signalParameterUpdated(const Object *, const Parameter *) override {}
  void signalParameterRemoved(const Object *, const Parameter *) override {}
  void signalArrayMapped(const Array *) override {}
  void signalArrayUnmapped(const Array *) override {}
  void signalObjectParameterUseCountZero(const Object *obj) override {};
  void signalObjectLayerUseCountZero(const Object *obj) override {};
  void signalObjectRemoved(const Object *) override {}
  void signalRemoveAllObjects() override {}
  void signalLayerAdded(const Layer *) override {}
  void signalLayerUpdated(const Layer *) override {}
  void signalLayerRemoved(const Layer *) override {}
  void signalActiveLayersChanged() override {}
  void signalObjectFilteringChanged() override {}
  void signalInvalidateCachedObjects() override {}
  void signalAnimationTimeChanged(float) override {}
};

/// Update delegate that dispatches signals to N held other update delegates
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
  void signalArrayMapped(const Array *a) override;
  void signalArrayUnmapped(const Array *a) override;
  void signalObjectParameterUseCountZero(const Object *obj) override;
  void signalObjectLayerUseCountZero(const Object *obj) override;
  void signalObjectRemoved(const Object *o) override;
  void signalRemoveAllObjects() override;
  void signalLayerAdded(const Layer *) override;
  void signalLayerUpdated(const Layer *) override;
  void signalLayerRemoved(const Layer *) override;
  void signalActiveLayersChanged() override;
  void signalObjectFilteringChanged() override;
  void signalInvalidateCachedObjects() override;
  void signalAnimationTimeChanged(float time) override;

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

} // namespace tsd::core