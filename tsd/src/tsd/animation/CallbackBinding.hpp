// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Binding.hpp"
// tsd_core
#include "tsd/core/TypeMacros.hpp"
// std
#include <functional>

namespace tsd::animation {

/*
 * Binding that invokes a user-supplied callback whenever the animation time
 * is updated. The callback receives only the current time; any scene objects
 * to modify should be captured directly in the closure.
 *
 * Example:
 *   auto *obj = scene.getObject(...);
 *   animation.addCallbackBinding([obj](float t) {
 *     float v = std::sin(t * 2.f * M_PI);
 *     obj->setParameter("opacity", ANARI_FLOAT32, &v);
 *   });
 */
struct CallbackBinding : public Binding
{
  TSD_DEFAULT_COPYABLE(CallbackBinding)
  TSD_DEFAULT_MOVEABLE(CallbackBinding)

  using Callback = std::function<void(float)>;

  CallbackBinding(Callback callback);

  void update(float t) override;

 private:
  Callback m_callback;
};

} // namespace tsd::animation
