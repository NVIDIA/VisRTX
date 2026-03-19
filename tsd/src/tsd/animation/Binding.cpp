// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/Binding.hpp"

namespace tsd::animation {

Binding::Binding(scene::Scene *scene) : m_scene(scene) {}

scene::Scene *Binding::scene() const
{
  return m_scene;
}

} // namespace tsd::animation
