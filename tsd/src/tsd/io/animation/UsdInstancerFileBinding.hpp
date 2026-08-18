// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Animation.hpp"
#include "tsd/io/animation/UsdFileBinding.hpp"
#include "tsd/scene/ObjectUsePtr.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Array.hpp"
// std
#include <memory>
#include <string>

namespace tsd::io {

/*
 * Binding that re-fills the transform Array of one point-instancer Prototype
 * from a Stage Session at the current animation time. It re-reads the
 * instancer at a Time Code and re-applies the same Prototype and visibility
 * selection the Import applied, then writes through the Array the Import
 * created -- it does not re-run conversion, which would churn object identity
 * and force ANARI handle teardown for half a million instances that only
 * moved.
 *
 * A mid-sequence change in placement count allocates a right-sized Array and
 * re-points the node at it, because a TSD Array cannot resize.
 *
 * Example:
 *   anim.emplaceFileBinding<UsdInstancerFileBinding>(
 *       &scene, session, arrayNode, transforms, file, "/root/points", 0);
 */
struct UsdInstancerFileBinding : public UsdFileBinding
{
  UsdInstancerFileBinding(scene::Scene *scene,
      std::shared_ptr<usd::UsdStageSession> session,
      scene::LayerNodeRef arrayNode,
      scene::ArrayRef transforms,
      std::string stageFile,
      std::string primPath,
      size_t prototypeIndex);
  ~UsdInstancerFileBinding() override;

  // FileBinding interface //

  std::string kind() const override;
  void toDataNode(tsd::core::DataNode &node) const override;
  std::vector<scene::LayerNodeRef> layerTargets() const override;
  void onDefragment(const scene::IndexRemapper &cb) override;

  void update(float t) override;

  // Reconstruct from a serialized node; returns null if the target layer node
  // is missing from the scene or is not a transform-array node.
  static UsdInstancerFileBinding *addToAnimation(tsd::animation::Animation &anim,
      scene::Scene &scene,
      tsd::core::DataNode &node);

 private:
  void addCallbackToAnimation(tsd::animation::Animation &anim) override;
  const char *logTag() const override;

  scene::LayerNodeRef m_arrayNode;
  scene::ObjectUsePtr<scene::Array, scene::Object::UseKind::ANIM> m_transforms;
  size_t m_prototypeIndex{0};
  bool m_sampleTimesNoted{false};
};

} // namespace tsd::io
