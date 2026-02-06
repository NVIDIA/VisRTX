// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Object.hpp"

namespace tsd::core {

struct Scene;

struct AnyObjectUsePtr
{
  AnyObjectUsePtr() = default;
  AnyObjectUsePtr(Object &o);
  AnyObjectUsePtr(const AnyObjectUsePtr &o);
  AnyObjectUsePtr(AnyObjectUsePtr &&o);
  ~AnyObjectUsePtr();

  AnyObjectUsePtr &operator=(const AnyObjectUsePtr &o);
  AnyObjectUsePtr &operator=(AnyObjectUsePtr &&o);
  AnyObjectUsePtr &operator=(Object &o);

  void reset();

  const Object *get() const;
  const Object *operator->() const;
  const Object &operator*() const;
  Object *get();
  Object *operator->();
  Object &operator*();

  template <typename T>
  const T *getAs() const;
  template <typename T>
  T *getAs();

  operator bool() const;

 private:
  Any m_object;
  Scene *m_scene{nullptr};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline const T *AnyObjectUsePtr::getAs() const
{
  static_assert(std::is_base_of<Object, T>::value,
      "AnyObjectUsePtr::getAs<T> requires T to derive from Object");
  return get() != nullptr ? dynamic_cast<const T *>(get()) : nullptr;
}

template <typename T>
T *AnyObjectUsePtr::getAs()
{
  static_assert(std::is_base_of<Object, T>::value,
      "AnyObjectUsePtr::getAs<T> requires T to derive from Object");
  return get() != nullptr ? dynamic_cast<T *>(get()) : nullptr;
}

} // namespace tsd::core
