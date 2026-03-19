// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tsd::scene {
struct Scene;
} // namespace tsd::scene

namespace tsd::animation {

/*
 * Base class for animation bindings that associates a binding with its owning
 * scene; derived types add target-specific data and interpolation logic.
 *
 * Example:
 *   struct MyBinding : Binding {
 *     MyBinding(scene::Scene *s) : Binding(s) {}
 *   };
 */
struct Binding
{
  Binding() = default;
  Binding(scene::Scene *scene);
  virtual ~Binding() = default;

  scene::Scene *scene() const;

 private:
  scene::Scene *m_scene{nullptr};
};

} // namespace tsd::animation
