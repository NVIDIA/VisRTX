// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/animation/SpatialFieldFileBinding.hpp"
#include "tsd/io/importers.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
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
    : FileBinding(scene),
      m_volume(volume),
      m_currentField(initialField),
      m_files(std::move(files))
{}

std::string SpatialFieldFileBinding::kind() const
{
  return "spatialField";
}

void SpatialFieldFileBinding::toDataNode(core::DataNode &node) const
{
  auto *vol = m_volume.get();
  node["targetIndex"] = vol ? vol->index() : size_t(-1);

  auto &filesNode = node["files"];
  for (const auto &f : m_files)
    filesNode.append() = f;
}

void SpatialFieldFileBinding::addCallbackToAnimation(
    tsd::animation::Animation &anim)
{
  anim.addCallbackBinding([this](float t) { update(t); });
}

void SpatialFieldFileBinding::onDefragment(const scene::IndexRemapper &cb)
{
  if (m_volume) {
    size_t newIdx = cb(m_volume->type(), m_volume->index());
    m_volume.updateDefragmentedIndex(newIdx);
  }
  if (m_currentField) {
    size_t newIdx = cb(m_currentField->type(), m_currentField->index());
    m_currentField.updateDefragmentedIndex(newIdx);
  }
}

void SpatialFieldFileBinding::update(float t)
{
  if (m_files.empty())
    return;

  const int N = static_cast<int>(m_files.size());
  const int idx = std::clamp(
      static_cast<int>(std::round(t * static_cast<float>(N - 1))), 0, N - 1);

  if (idx == m_currentFrame)
    return;

  // Load new field before removing old one to avoid a brief gap
  auto newField = import_spatial_field(*scene(), m_files[idx].c_str());
  if (!newField) {
    logWarning("[SpatialFieldFileBinding] failed to load frame %d: '%s'",
        idx,
        m_files[idx].c_str());
    return;
  }

  // Swap the Volume's spatial field
  if (auto *vol = m_volume.get(); vol != nullptr)
    vol->setParameterObject("value", *newField);

  m_currentField = newField;
  m_currentFrame = idx;

  scene()->removeUnusedObjects();
}

size_t SpatialFieldFileBinding::frameCount() const
{
  return m_files.size();
}

int SpatialFieldFileBinding::currentFrame() const
{
  return m_currentFrame;
}

} // namespace tsd::io
