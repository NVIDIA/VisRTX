// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/scene/Object.hpp"
#include "tsd/scene/Scene.hpp"

namespace tsd::scene {

struct Scene;

/*
 * Type-erased owning pointer to any Object subclass that increments the
 * object's APP use count on assignment and decrements it on release.
 *
 * Example:
 *   AnyObjectUsePtr ptr(someObject);
 *   auto *geom = ptr.getAs<Geometry>();
 *   ptr.reset();
 */
template <Object::UseKind K = Object::UseKind::APP>
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

  // WARNING: this is just for dealing with scene defragmentation, do not use
  // directly!
  void updateDefragmentedIndex(size_t newIndex);

 private:
  Any m_object;
  Scene *m_scene{nullptr};
};

template <Object::UseKind K>
bool operator==(const AnyObjectUsePtr<K> &a, const AnyObjectUsePtr<K> &b);
template <Object::UseKind K>
bool operator!=(const AnyObjectUsePtr<K> &a, const AnyObjectUsePtr<K> &b);

// Inlined definitions ////////////////////////////////////////////////////////

template <Object::UseKind K>
inline AnyObjectUsePtr<K>::AnyObjectUsePtr(Object &o)
{
  if (o.scene() != nullptr) {
    m_scene = o.scene();
    m_object = Any(o.type(), o.index());
    o.incUseCount(K);
  }
}

template <Object::UseKind K>
inline AnyObjectUsePtr<K>::AnyObjectUsePtr(const AnyObjectUsePtr &o)
{
  m_object = o.m_object;
  m_scene = o.m_scene;
  if (auto *obj = get(); obj != nullptr)
    obj->incUseCount(K);
}

template <Object::UseKind K>
inline AnyObjectUsePtr<K>::AnyObjectUsePtr(AnyObjectUsePtr &&o)
{
  m_object = std::move(o.m_object);
  m_scene = o.m_scene;
  o.m_scene = nullptr;
  o.m_object.reset();
}

template <Object::UseKind K>
inline AnyObjectUsePtr<K>::~AnyObjectUsePtr()
{
  reset();
}

template <Object::UseKind K>
inline AnyObjectUsePtr<K> &AnyObjectUsePtr<K>::operator=(
    const AnyObjectUsePtr &o)
{
  if (this != &o) {
    reset();
    m_object = o.m_object;
    m_scene = o.m_scene;
    if (auto *obj = get(); obj != nullptr)
      obj->incUseCount(K);
  }
  return *this;
}

template <Object::UseKind K>
inline AnyObjectUsePtr<K> &AnyObjectUsePtr<K>::operator=(AnyObjectUsePtr &&o)
{
  m_object = std::move(o.m_object);
  m_scene = o.m_scene;
  o.m_scene = nullptr;
  o.m_object.reset();
  return *this;
}

template <Object::UseKind K>
inline AnyObjectUsePtr<K> &AnyObjectUsePtr<K>::operator=(Object &o)
{
  reset();
  if (o.scene() != nullptr) {
    m_scene = o.scene();
    m_object = Any(o.type(), o.index());
    o.incUseCount(K);
  }
  return *this;
}

template <Object::UseKind K>
inline void AnyObjectUsePtr<K>::reset()
{
  if (auto *obj = get(); obj != nullptr)
    obj->decUseCount(K);

  m_scene = nullptr;
}

template <Object::UseKind K>
inline const Object *AnyObjectUsePtr<K>::get() const
{
  return m_scene && m_object ? m_scene->getObject(m_object) : nullptr;
}

template <Object::UseKind K>
inline const Object *AnyObjectUsePtr<K>::operator->() const
{
  return get();
}

template <Object::UseKind K>
inline const Object &AnyObjectUsePtr<K>::operator*() const
{
  return *get();
}

template <Object::UseKind K>
inline Object *AnyObjectUsePtr<K>::get()
{
  return m_scene && m_object ? m_scene->getObject(m_object) : nullptr;
}

template <Object::UseKind K>
inline Object *AnyObjectUsePtr<K>::operator->()
{
  return get();
}

template <Object::UseKind K>
inline Object &AnyObjectUsePtr<K>::operator*()
{
  return *get();
}

template <Object::UseKind K>
inline AnyObjectUsePtr<K>::operator bool() const
{
  return get() != nullptr;
}

template <Object::UseKind K>
inline bool operator==(const AnyObjectUsePtr<K> &a, const AnyObjectUsePtr<K> &b)
{
  auto *a1 = a.get();
  auto *b1 = b.get();
  return (a1 && b1) && (a1->type() == b1->type())
      && (a1->index() == b1->index());
}

template <Object::UseKind K>
inline bool operator!=(const AnyObjectUsePtr<K> &a, const AnyObjectUsePtr<K> &b)
{
  return !(a == b);
}

template <Object::UseKind K>
template <typename T>
inline const T *AnyObjectUsePtr<K>::getAs() const
{
  static_assert(std::is_base_of<Object, T>::value,
      "AnyObjectUsePtr::getAs<T> requires T to derive from Object");
  return get() != nullptr ? dynamic_cast<const T *>(get()) : nullptr;
}

template <Object::UseKind K>
template <typename T>
T *AnyObjectUsePtr<K>::getAs()
{
  static_assert(std::is_base_of<Object, T>::value,
      "AnyObjectUsePtr::getAs<T> requires T to derive from Object");
  return get() != nullptr ? dynamic_cast<T *>(get()) : nullptr;
}

template <Object::UseKind K>
void AnyObjectUsePtr<K>::updateDefragmentedIndex(size_t newIndex)
{
  if (!m_scene || !m_object)
    return;
  m_object = Any(m_object.type(), newIndex);
}

} // namespace tsd::scene
