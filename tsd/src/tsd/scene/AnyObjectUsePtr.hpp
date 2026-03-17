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

bool operator==(const AnyObjectUsePtr &a, const AnyObjectUsePtr &b);
bool operator!=(const AnyObjectUsePtr &a, const AnyObjectUsePtr &b);

// Inlined definitions ////////////////////////////////////////////////////////

inline AnyObjectUsePtr::AnyObjectUsePtr(Object &o)
{
  reset();
}

inline AnyObjectUsePtr::AnyObjectUsePtr(const AnyObjectUsePtr &o)
{
  m_object = o.m_object;
  m_scene = o.m_scene;
  if (auto *obj = get(); obj != nullptr)
    obj->incUseCount(Object::UseKind::APP);
}

inline AnyObjectUsePtr::AnyObjectUsePtr(AnyObjectUsePtr &&o)
{
  m_object = std::move(o.m_object);
  m_scene = o.m_scene;
  o.m_scene = nullptr;
  o.m_object.reset();
}

inline AnyObjectUsePtr::~AnyObjectUsePtr()
{
  reset();
}

inline AnyObjectUsePtr &AnyObjectUsePtr::operator=(const AnyObjectUsePtr &o)
{
  if (this != &o) {
    reset();
    m_object = o.m_object;
    m_scene = o.m_scene;
    if (auto *obj = get(); obj != nullptr)
      obj->incUseCount(Object::UseKind::APP);
  }
  return *this;
}

inline AnyObjectUsePtr &AnyObjectUsePtr::operator=(AnyObjectUsePtr &&o)
{
  m_object = std::move(o.m_object);
  m_scene = o.m_scene;
  o.m_scene = nullptr;
  o.m_object.reset();
  return *this;
}

inline AnyObjectUsePtr &AnyObjectUsePtr::operator=(Object &o)
{
  reset();
  if (o.scene() != nullptr) {
    m_scene = o.scene();
    m_object = Any(o.type(), o.index());
    o.incUseCount(Object::UseKind::APP);
  }
  return *this;
}

inline void AnyObjectUsePtr::reset()
{
  if (auto *obj = get(); obj != nullptr)
    obj->decUseCount(Object::UseKind::APP);

  m_scene = nullptr;
}

inline const Object *AnyObjectUsePtr::get() const
{
  return m_scene && m_object ? m_scene->getObject(m_object) : nullptr;
}

inline const Object *AnyObjectUsePtr::operator->() const
{
  return get();
}

inline const Object &AnyObjectUsePtr::operator*() const
{
  return *get();
}

inline Object *AnyObjectUsePtr::get()
{
  return m_scene && m_object ? m_scene->getObject(m_object) : nullptr;
}

inline Object *AnyObjectUsePtr::operator->()
{
  return get();
}

inline Object &AnyObjectUsePtr::operator*()
{
  return *get();
}

inline AnyObjectUsePtr::operator bool() const
{
  return get() != nullptr;
}

inline bool operator==(const AnyObjectUsePtr &a, const AnyObjectUsePtr &b)
{
  auto *a1 = a.get();
  auto *b1 = b.get();
  return (a1 && b1) && (a1->type() == b1->type())
      && (a1->index() == b1->index());
}

inline bool operator!=(const AnyObjectUsePtr &a, const AnyObjectUsePtr &b)
{
  return !(a == b);
}

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

} // namespace tsd::scene
