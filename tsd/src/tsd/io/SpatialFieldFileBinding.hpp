// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/Binding.hpp"
#include "tsd/scene/ObjectUsePtr.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/SpatialField.hpp"
#include "tsd/scene/objects/Volume.hpp"
// std
#include <memory>
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
 *   SpatialFieldFileBinding b(&scene, vol.data(), field0, files);
 *   b.addToAnimation(anim);
 *
 *   // Or use the higher-level helper:
 *   import_volume_animation(scene, animMgr, files, location);
 */
struct SpatialFieldFileBinding : public tsd::animation::Binding
{
  // Construct the binding.  `initialField` must already be set as the "value"
  // parameter on `volume` and corresponds to files[0].
  SpatialFieldFileBinding(scene::Scene *scene,
      scene::Volume *volume,
      scene::SpatialFieldRef initialField,
      std::vector<std::string> files);

  // Registers a CallbackBinding on `anim` that calls invoke() on each time
  // change.  The binding state is reference-counted via shared_ptr, so this
  // SpatialFieldFileBinding object may be destroyed after addToAnimation().
  void addToAnimation(tsd::animation::Animation &anim);

  // Load the appropriate file for time t.  No-ops if the frame has not changed.
  void invoke(float t);

  size_t frameCount() const;
  int currentFrame() const;

 private:
  struct SharedState
  {
    scene::Scene *scene{nullptr};
    scene::ObjectUsePtr<scene::Volume, scene::Object::UseKind::ANIM> volume;
    scene::SpatialFieldRef currentField;
    std::vector<std::string> files;
    int currentFrame{0};
  };

  std::shared_ptr<SharedState> m_state;
};

} // namespace tsd::io
