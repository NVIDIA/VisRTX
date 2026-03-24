// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/SpatialFieldFileBinding.hpp"
#include "tsd/io/importers.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// std
#include <algorithm>
#include <cmath>

namespace tsd::io {

using namespace tsd::core;

SpatialFieldFileBinding::SpatialFieldFileBinding(scene::Scene *scene,
    scene::Volume *volume,
    scene::SpatialFieldRef initialField,
    std::vector<std::string> files)
    : Binding(scene), m_state(std::make_shared<SharedState>())
{
  m_state->scene = scene;
  m_state->volume = volume;
  m_state->currentField = initialField;
  m_state->files = std::move(files);
  m_state->currentFrame = 0;
}

void SpatialFieldFileBinding::addToAnimation(tsd::animation::Animation &anim)
{
  auto state = m_state; // capture by value (shared_ptr)
  anim.addCallbackBinding([state](float t) {
    if (state->files.empty())
      return;

    const int N = static_cast<int>(state->files.size());
    const int idx = std::clamp(
        static_cast<int>(std::round(t * static_cast<float>(N - 1))), 0, N - 1);

    if (idx == state->currentFrame)
      return;

    // Load new field before removing old one to avoid a brief gap
    auto newField =
        import_spatial_field(*state->scene, state->files[idx].c_str());
    if (!newField) {
      logWarning("[SpatialFieldFileBinding] failed to load frame %d: '%s'",
          idx,
          state->files[idx].c_str());
      return;
    }

    // Swap the Volume's spatial field
    auto *vol = state->volume.get();
    if (vol) {
      vol->setParameterObject("value", *newField);
      auto range = newField->computeValueRange();
      vol->setParameter("valueRange", ANARI_FLOAT32_BOX1, &range);
    }

    // Remove the old SpatialField from the scene now that it's no longer
    // referenced by the Volume's "value" parameter
    if (state->currentField)
      state->scene->removeObject(state->currentField.data());

    state->currentField = newField;
    state->currentFrame = idx;
  });
}

void SpatialFieldFileBinding::invoke(float t)
{
  // Replicate the callback logic for direct invocation
  if (!m_state || m_state->files.empty())
    return;

  const int N = static_cast<int>(m_state->files.size());
  const int idx = std::clamp(
      static_cast<int>(std::round(t * static_cast<float>(N - 1))), 0, N - 1);

  if (idx == m_state->currentFrame)
    return;

  auto newField =
      import_spatial_field(*m_state->scene, m_state->files[idx].c_str());
  if (!newField) {
    logWarning("[SpatialFieldFileBinding] failed to load frame %d: '%s'",
        idx,
        m_state->files[idx].c_str());
    return;
  }

  auto *vol = m_state->volume.get();
  if (vol) {
    vol->setParameterObject("value", *newField);
    auto range = newField->computeValueRange();
    vol->setParameter("valueRange", ANARI_FLOAT32_BOX1, &range);
  }

  if (m_state->currentField)
    m_state->scene->removeObject(m_state->currentField.data());

  m_state->currentField = newField;
  m_state->currentFrame = idx;
}

size_t SpatialFieldFileBinding::frameCount() const
{
  return m_state ? m_state->files.size() : 0u;
}

int SpatialFieldFileBinding::currentFrame() const
{
  return m_state ? m_state->currentFrame : -1;
}

} // namespace tsd::io
