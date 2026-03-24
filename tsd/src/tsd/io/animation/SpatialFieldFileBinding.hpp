// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/FileBinding.hpp"
#include "tsd/scene/ObjectUsePtr.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/SpatialField.hpp"
#include "tsd/scene/objects/Volume.hpp"
// std
#include <string>
#include <vector>

namespace tsd::io {

/*
 * Binding that drives a Volume's spatial field by loading a different file at
 * each animation time step.  Time t=0.0 selects files[0]; t=1.0 selects
 * files[N-1].  Only the currently-visible frame's SpatialField is held in
 * memory; the previous frame's field is removed from the scene on each step.
 *
 * Example:
 *   auto field0 = import_spatial_field(scene, files[0].c_str());
 *   auto vol = ...; // volume with field0 as "value"
 *   auto &b = anim.emplaceFileBinding<SpatialFieldFileBinding>(&scene,
 * vol.data(), field0, files); b.addCallbackToAnimation(anim);
 *
 *   // Or use the higher-level helper:
 *   import_volume_animation(scene, animMgr, files, location);
 */
struct SpatialFieldFileBinding : public tsd::animation::FileBinding
{
  // Construct the binding.  `initialField` must already be set as the "value"
  // parameter on `volume` and corresponds to files[0].
  SpatialFieldFileBinding(scene::Scene *scene,
      scene::Volume *volume,
      scene::SpatialFieldRef initialField,
      std::vector<std::string> files);

  // FileBinding interface //

  std::string kind() const override;

  // Writes targetIndex (the volume's object-pool index) and the file list.
  void toDataNode(tsd::core::DataNode &node) const override;

  // Registers a CallbackBinding on `anim` that loads the appropriate file on
  // each time change.  Does NOT add this binding to anim.fileBindings().
  // Called both during initial import and during reconstruction from a
  // DataNode.
  void addCallbackToAnimation(tsd::animation::Animation &anim) override;

  void onDefragment(const scene::IndexRemapper &cb) override;

  // Load the appropriate file for time t.  No-ops if the frame has not changed.
  void update(float t) override;

  size_t frameCount() const;
  int currentFrame() const;

 private:
  scene::ObjectUsePtr<scene::Volume, scene::Object::UseKind::ANIM> m_volume;
  scene::ObjectUsePtr<scene::SpatialField, scene::Object::UseKind::ANIM>
      m_currentField;
  std::vector<std::string> m_files;
  int m_currentFrame{0};
};

} // namespace tsd::io
