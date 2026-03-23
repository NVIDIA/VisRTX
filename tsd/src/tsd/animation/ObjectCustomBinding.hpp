// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Binding.hpp"
// tsd_core
#include "tsd/core/TypeMacros.hpp"
// tsd_scene
#include "tsd/scene/AnyObjectUsePtr.hpp"
// std
#include <functional>

namespace tsd::animation {

/*
 * Binding that invokes a user-supplied callback on a target scene object
 * whenever the animation time is updated. The callback receives both the
 * target object and the current time, allowing arbitrary procedural behavior.
 *
 * Example:
 *   animation.addCustomObjectBinding(obj,
 *       [](scene::Object &o, float t) {
 *         float v = std::sin(t * 2.f * M_PI);
 *         o.setParameter("opacity", ANARI_FLOAT32, &v);
 *       });
 */
struct ObjectCustomBinding : public Binding
{
  TSD_DEFAULT_COPYABLE(ObjectCustomBinding)
  TSD_DEFAULT_MOVEABLE(ObjectCustomBinding)

  using Callback = std::function<void(scene::Object &, float)>;

  ObjectCustomBinding(scene::Object *target, Callback callback);

  scene::Object *target() const;

  void invoke(float time) const;

 private:
  scene::AnyObjectUsePtr<scene::Object::UseKind::ANIM> m_target;
  Callback m_callback;
};

} // namespace tsd::animation
