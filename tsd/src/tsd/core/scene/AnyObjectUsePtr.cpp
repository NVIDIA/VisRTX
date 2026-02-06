// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/AnyObjectUsePtr.hpp"
#include "tsd/core/scene/Scene.hpp"

namespace tsd::core {

AnyObjectUsePtr::AnyObjectUsePtr(Object &o)
{
  reset();
}

AnyObjectUsePtr::AnyObjectUsePtr(const AnyObjectUsePtr &o)
{
  m_object = o.m_object;
  m_scene = o.m_scene;
  if (auto *obj = get(); obj != nullptr)
    obj->incUseCount(Object::UseKind::APP);
}

AnyObjectUsePtr::AnyObjectUsePtr(AnyObjectUsePtr &&o)
{
  m_object = std::move(o.m_object);
  m_scene = o.m_scene;
  o.m_scene = nullptr;
  o.m_object.reset();
}

AnyObjectUsePtr::~AnyObjectUsePtr()
{
  reset();
}

AnyObjectUsePtr &AnyObjectUsePtr::operator=(const AnyObjectUsePtr &o)
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

AnyObjectUsePtr &AnyObjectUsePtr::operator=(AnyObjectUsePtr &&o)
{
  m_object = std::move(o.m_object);
  m_scene = o.m_scene;
  o.m_scene = nullptr;
  o.m_object.reset();
  return *this;
}

AnyObjectUsePtr &AnyObjectUsePtr::operator=(Object &o)
{
  reset();
  if (o.scene() != nullptr) {
    m_scene = o.scene();
    m_object = Any(o.type(), o.index());
    o.incUseCount(Object::UseKind::APP);
  }
  return *this;
}

void AnyObjectUsePtr::reset()
{
  if (auto *obj = get(); obj != nullptr)
    obj->decUseCount(Object::UseKind::APP);

  m_scene = nullptr;
}

const Object *AnyObjectUsePtr::get() const
{
  return m_scene && m_object ? m_scene->getObject(m_object) : nullptr;
}

const Object *AnyObjectUsePtr::operator->() const
{
  return get();
}

const Object &AnyObjectUsePtr::operator*() const
{
  return *get();
}

Object *AnyObjectUsePtr::get()
{
  return m_scene && m_object ? m_scene->getObject(m_object) : nullptr;
}

Object *AnyObjectUsePtr::operator->()
{
  return get();
}

Object &AnyObjectUsePtr::operator*()
{
  return *get();
}

AnyObjectUsePtr::operator bool() const
{
  return get() != nullptr;
}

} // namespace tsd::core
