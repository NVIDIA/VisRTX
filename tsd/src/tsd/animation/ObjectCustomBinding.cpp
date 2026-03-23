// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/ObjectCustomBinding.hpp"
// tsd_scene
#include "tsd/scene/Object.hpp"

namespace tsd::animation {

ObjectCustomBinding::ObjectCustomBinding(
    scene::Object *target, Callback callback)
    : Binding(target->scene()), m_callback(std::move(callback))
{
  if (target)
    m_target = *target;
}

scene::Object *ObjectCustomBinding::target() const
{
  return m_target.get();
}

void ObjectCustomBinding::invoke(float time) const
{
  if (m_target && m_callback)
    m_callback(*m_target, time);
}

} // namespace tsd::animation
