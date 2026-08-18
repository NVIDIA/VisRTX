// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Animation.hpp"
#include "tsd/io/animation/UsdFileBinding.hpp"
#include "tsd/scene/ObjectUsePtr.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Geometry.hpp"
// tsd_core
#include "tsd/core/TSDMath.hpp"
// tsd_io
#include "tsd/io/usd/UsdGeometryResolveOptions.h"
// std
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tsd::io {

/*
 * Binding that re-resolves a deforming gprim from a Stage Session at the
 * current animation time and writes the result over the Geometries the Import
 * built, so that a long animation of a dense mesh does not have to fit in
 * memory (ADR 0018).
 *
 * It re-runs the resolve half of conversion and none of the build half: points,
 * indices and primvars arrive as one consistent set, while the Surfaces and
 * Materials around them keep their identity and their ANARI handles (ADR 0022).
 * Everything the Import decided that does not change over time -- which primvar
 * each Part's material reads, whether the mesh refines, what transform is baked
 * in -- is carried here and replayed rather than recomputed.
 *
 * The animation time maps onto the Stage's own Time Code range and USD
 * evaluates there, so a value between authored samples is interpolated rather
 * than snapped -- the same thing usdview shows.
 *
 * Example:
 *   auto &b = anim.emplaceFileBinding<UsdGeometryFileBinding>(
 *       &scene, session, stageFile, "/World/Character", converted);
 */
struct UsdGeometryFileBinding : public UsdFileBinding
{
  // One resolved Part and the Geometry it was built into.
  struct Part
  {
    std::string name;
    scene::ObjectUsePtr<scene::Geometry, scene::Object::UseKind::ANIM> geometry;
  };

  UsdGeometryFileBinding(scene::Scene *scene,
      std::shared_ptr<usd::UsdStageSession> session,
      std::string stageFile,
      std::string primPath,
      std::vector<Part> parts,
      usd::GeometryResolveOptions resolveOptions);
  ~UsdGeometryFileBinding() override;

  // FileBinding interface //

  std::string kind() const override;
  void toDataNode(tsd::core::DataNode &node) const override;
  void onDefragment(const scene::IndexRemapper &cb) override;

  // Re-resolve the gprim at the Time Code `t` maps to and write it over the
  // bound Geometries.
  void update(float t) override;

  // Reconstruct from a serialized node; returns null if no target geometry
  // survives in the scene.
  static UsdGeometryFileBinding *addToAnimation(tsd::animation::Animation &anim,
      scene::Scene &scene,
      tsd::core::DataNode &node);

 private:
  void addCallbackToAnimation(tsd::animation::Animation &anim) override;
  const char *logTag() const override;

  std::vector<Part> m_parts;
  usd::GeometryResolveOptions m_resolveOptions;
  bool m_sampleTimesNoted{false};
  bool m_partsChangedReported{false};
};

} // namespace tsd::io
