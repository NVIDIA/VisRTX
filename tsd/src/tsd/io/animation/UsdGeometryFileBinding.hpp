// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/FileBinding.hpp"
#include "tsd/scene/ObjectUsePtr.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Geometry.hpp"
// std
#include <string>
#include <vector>

namespace tsd::io {

/*
 * Binding that re-pulls a deforming mesh's vertex arrays from a USD Stage at
 * the current animation time, so that a long animation of a dense mesh does
 * not have to fit in memory. The Stage stays open for the binding's lifetime
 * (ADR 0018); serialization records the file and prim paths and reconstructs
 * by re-opening.
 *
 * Example:
 *   auto &b = anim.emplaceFileBinding<UsdGeometryFileBinding>(
 *       &scene, geometry.data(), stageFile, "/World/Character", sampleTimes);
 */
struct UsdGeometryFileBinding : public tsd::animation::FileBinding
{
  UsdGeometryFileBinding(scene::Scene *scene,
      scene::Geometry *geometry,
      std::string stageFile,
      std::string primPath,
      std::vector<double> sampleTimes);

  // FileBinding interface //

  std::string kind() const override;
  void toDataNode(tsd::core::DataNode &node) const override;
  void onDefragment(const scene::IndexRemapper &cb) override;

  // Pull the vertex arrays for the sample nearest `t`, which is normalized
  // across the binding's authored sample times. No-ops if the frame has not
  // changed.
  void update(float t) override;

  size_t frameCount() const;
  int currentFrame() const;

  // Reconstruct from a serialized node; returns null if the target geometry is
  // missing from the scene.
  static UsdGeometryFileBinding *addToAnimation(tsd::animation::Animation &anim,
      scene::Scene &scene,
      tsd::core::DataNode &node);

 private:
  void addCallbackToAnimation(tsd::animation::Animation &anim) override;

  scene::ObjectUsePtr<scene::Geometry, scene::Object::UseKind::ANIM> m_geometry;
  std::string m_stageFile;
  std::string m_primPath;
  std::vector<double> m_sampleTimes;
  int m_currentFrame{0};

  // Opened lazily on the first update and retained thereafter, so scrubbing
  // does not pay stage-open cost per frame.
  struct StageHolder;
  std::shared_ptr<StageHolder> m_stage;
};

} // namespace tsd::io
