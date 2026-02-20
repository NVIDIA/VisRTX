// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Object.hpp"

namespace tsd::core {

template <typename T, Object::UseKind K = Object::UseKind::APP>
struct ObjectUsePtr
{
  static_assert(isObject<T>(),
      "ObjectUsePtr can only be instantiated with tsd::core::Object types");

  ObjectUsePtr() = default;
  ObjectUsePtr(T *o);
  ObjectUsePtr(ObjectPoolRef<T> o);
  ObjectUsePtr(const ObjectUsePtr<T, K> &o);
  ObjectUsePtr(ObjectUsePtr<T, K> &&o);
  ~ObjectUsePtr();

  ObjectUsePtr &operator=(const ObjectUsePtr<T, K> &o);
  ObjectUsePtr &operator=(ObjectUsePtr<T, K> &&o);

  ObjectUsePtr &operator=(T *o);
  ObjectUsePtr &operator=(ObjectPoolRef<T> o);

  void reset();

  const T *get() const;
  const T *operator->() const;
  const T &operator*() const;
  T *get();
  T *operator->();
  T &operator*();

  operator bool() const;

 private:
  ObjectPoolRef<T> m_object;
};

template <typename T, Object::UseKind K>
bool operator==(const ObjectUsePtr<T, K> &a, const ObjectUsePtr<T, K> &b);

template <typename T, Object::UseKind K>
bool operator!=(const ObjectUsePtr<T, K> &a, const ObjectUsePtr<T, K> &b);

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T, Object::UseKind K>
inline ObjectUsePtr<T, K>::ObjectUsePtr(T *o)
    : m_object(o ? o->self() : ObjectPoolRef<T>{})
{
  if (m_object)
    m_object->incUseCount(Object::UseKind::APP);
}

template <typename T, Object::UseKind K>
inline ObjectUsePtr<T, K>::ObjectUsePtr(ObjectPoolRef<T> o) : m_object(o)
{
  if (m_object)
    m_object->incUseCount(Object::UseKind::APP);
}

template <typename T, Object::UseKind K>
inline ObjectUsePtr<T, K>::ObjectUsePtr(const ObjectUsePtr<T, K> &o)
    : m_object(o.m_object)
{
  if (m_object)
    m_object->incUseCount(Object::UseKind::APP);
}

template <typename T, Object::UseKind K>
inline ObjectUsePtr<T, K>::ObjectUsePtr(ObjectUsePtr<T, K> &&o)
    : m_object(o.m_object)
{
  o.m_object = {};
}

template <typename T, Object::UseKind K>
inline ObjectUsePtr<T, K>::~ObjectUsePtr()
{
  reset();
}

template <typename T, Object::UseKind K>
inline ObjectUsePtr<T, K> &ObjectUsePtr<T, K>::operator=(const ObjectUsePtr &o)
{
  if (this != &o) {
    reset();
    m_object = o.m_object;
    if (m_object)
      m_object->incUseCount(K);
  }
  return *this;
}

template <typename T, Object::UseKind K>
inline ObjectUsePtr<T, K> &ObjectUsePtr<T, K>::operator=(ObjectUsePtr<T, K> &&o)
{
  if (this != &o) {
    reset();
    m_object = o.m_object;
    o.m_object = {};
    if (m_object)
      m_object->incUseCount(K);
  }
  return *this;
}

template <typename T, Object::UseKind K>
inline ObjectUsePtr<T, K> &ObjectUsePtr<T, K>::operator=(T *o)
{
  reset();
  if (o) {
    m_object = o->self();
    o->incUseCount(K);
  }
  return *this;
}

template <typename T, Object::UseKind K>
inline ObjectUsePtr<T, K> &ObjectUsePtr<T, K>::operator=(ObjectPoolRef<T> o)
{
  static_assert(isObject<T>(),
      "ObjectUsePtr can only be assigned ObjectPoolRef<T> when T is a"
      " tsd::core::Object type");
  reset();
  if (o) {
    m_object = o;
    o->incUseCount(K);
  }
  return *this;
}

template <typename T, Object::UseKind K>
void ObjectUsePtr<T, K>::reset()
{
  if (m_object)
    m_object->decUseCount(K);
  m_object = {};
}

template <typename T, Object::UseKind K>
inline const T *ObjectUsePtr<T, K>::get() const
{
  return m_object.data();
}

template <typename T, Object::UseKind K>
inline const T *ObjectUsePtr<T, K>::operator->() const
{
  return m_object.data();
}

template <typename T, Object::UseKind K>
inline const T &ObjectUsePtr<T, K>::operator*() const
{
  return *get();
}

template <typename T, Object::UseKind K>
inline T *ObjectUsePtr<T, K>::get()
{
  return m_object.data();
}

template <typename T, Object::UseKind K>
inline T *ObjectUsePtr<T, K>::operator->()
{
  return m_object.data();
}

template <typename T, Object::UseKind K>
inline T &ObjectUsePtr<T, K>::operator*()
{
  return *get();
}

template <typename T, Object::UseKind K>
inline ObjectUsePtr<T, K>::operator bool() const
{
  return m_object;
}

template <typename T, Object::UseKind K>
inline bool operator==(const ObjectUsePtr<T, K> &a, const ObjectUsePtr<T, K> &b)
{
  auto *a1 = a.get();
  auto *b1 = b.get();
  return (a1 && b1) && (a1->type() == b1->type())
      && (a1->index() == b1->index());
}

template <typename T, Object::UseKind K>
inline bool operator!=(const ObjectUsePtr<T, K> &a, const ObjectUsePtr<T, K> &b)
{
  return !(a == b);
}

} // namespace tsd::core
