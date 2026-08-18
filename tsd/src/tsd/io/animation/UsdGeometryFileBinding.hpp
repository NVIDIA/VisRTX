// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Animation.hpp"
#include "tsd/io/animation/UsdFileBinding.hpp"
#include "tsd/scene/ObjectUsePtr.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Geometry.hpp"
// std
#include <memory>
#include <string>

namespace tsd::io {

/*
 * Binding that re-pulls a deforming mesh's vertex arrays from a Stage Session
 * at the current animation time, so that a long animation of a dense mesh does
 * not have to fit in memory. The Session stays open for the binding's lifetime
 * (ADR 0018); serialization records the file and prim paths and reconstructs
 * by rejoining the Session for that file.
 *
 * The animation time maps onto the Stage's own Time Code range and USD
 * evaluates there, so a value between authored samples is interpolated rather
 * than snapped -- the same thing usdview shows.
 *
 * Example:
 *   auto &b = anim.emplaceFileBinding<UsdGeometryFileBinding>(
 *       &scene, geometry.data(), session, stageFile, "/World/Character");
 */
struct UsdGeometryFileBinding : public UsdFileBinding
{
  UsdGeometryFileBinding(scene::Scene *scene,
      scene::Geometry *geometry,
      std::shared_ptr<usd::UsdStageSession> session,
      std::string stageFile,
      std::string primPath);
  ~UsdGeometryFileBinding() override;

  // FileBinding interface //

  std::string kind() const override;
  void toDataNode(tsd::core::DataNode &node) const override;
  void onDefragment(const scene::IndexRemapper &cb) override;

  // Pull the vertex arrays for the Time Code `t` maps to.
  void update(float t) override;

  // Reconstruct from a serialized node; returns null if the target geometry is
  // missing from the scene.
  static UsdGeometryFileBinding *addToAnimation(tsd::animation::Animation &anim,
      scene::Scene &scene,
      tsd::core::DataNode &node);

 private:
  void addCallbackToAnimation(tsd::animation::Animation &anim) override;
  const char *logTag() const override;

  scene::ObjectUsePtr<scene::Geometry, scene::Object::UseKind::ANIM> m_geometry;
  bool m_sampleTimesNoted{false};
  bool m_countChangeReported{false};
};

} // namespace tsd::io
