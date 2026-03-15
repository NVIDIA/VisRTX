// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/scene/Object.hpp"
#include "tsd/scene/Scene.hpp"

namespace tsd::animation {

// RAII handle for animation binding targets.
// Stores (type, index, scene) and manages UseKind::ANIM reference counting.
// Survives defragmentation via index remapping.
struct AnimObjectRef
{
  AnimObjectRef() = default;

  explicit AnimObjectRef(tsd::scene::Object &obj)
      : m_type(obj.type()), m_index(obj.index()), m_scene(obj.scene())
  {
    obj.incUseCount(tsd::scene::Object::UseKind::ANIM);
  }

  ~AnimObjectRef()
  {
    decRef();
  }

  AnimObjectRef(const AnimObjectRef &o)
      : m_type(o.m_type), m_index(o.m_index), m_scene(o.m_scene)
  {
    incRef();
  }

  AnimObjectRef &operator=(const AnimObjectRef &o)
  {
    if (this != &o) {
      decRef();
      m_type = o.m_type;
      m_index = o.m_index;
      m_scene = o.m_scene;
      incRef();
    }
    return *this;
  }

  AnimObjectRef(AnimObjectRef &&o) noexcept
      : m_type(o.m_type), m_index(o.m_index), m_scene(o.m_scene)
  {
    o.m_type = ANARI_UNKNOWN;
    o.m_index = tsd::core::INVALID_INDEX;
    o.m_scene = nullptr;
  }

  AnimObjectRef &operator=(AnimObjectRef &&o) noexcept
  {
    if (this != &o) {
      decRef();
      m_type = o.m_type;
      m_index = o.m_index;
      m_scene = o.m_scene;
      o.m_type = ANARI_UNKNOWN;
      o.m_index = tsd::core::INVALID_INDEX;
      o.m_scene = nullptr;
    }
    return *this;
  }

  tsd::scene::Object *resolve() const
  {
    return m_scene ? m_scene->getObject(m_type, m_index) : nullptr;
  }

  operator bool() const
  {
    return m_type != ANARI_UNKNOWN && m_index != tsd::core::INVALID_INDEX;
  }

  ANARIDataType type() const { return m_type; }
  size_t index() const { return m_index; }

  void updateIndex(size_t newIndex) { m_index = newIndex; }

 private:
  void incRef()
  {
    if (auto *obj = resolve())
      obj->incUseCount(tsd::scene::Object::UseKind::ANIM);
  }

  void decRef()
  {
    if (auto *obj = resolve())
      obj->decUseCount(tsd::scene::Object::UseKind::ANIM);
  }

  ANARIDataType m_type{ANARI_UNKNOWN};
  size_t m_index{tsd::core::INVALID_INDEX};
  tsd::scene::Scene *m_scene{nullptr};
};

} // namespace tsd::animation
