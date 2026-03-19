// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Binding.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
#include "tsd/core/TypeMacros.hpp"
// tsd_scene
#include "tsd/scene/LayerNodeData.hpp"
// std
#include <vector>

namespace tsd::animation {

/*
 * Binding that animates a scene-graph node's transform over time by
 * interpolating decomposed rotation, translation, and scale keyframes.
 *
 * Example:
 *   TransformBinding tb(nodeRef, times, rotations, translations, scales, n);
 *   const math::float4 &r = tb.rotation(0);
 *   tb.insertKeyframe(0.5f, newTransform);
 */
struct TransformBinding : public Binding
{
  TSD_DEFAULT_COPYABLE(TransformBinding)
  TSD_DEFAULT_MOVEABLE(TransformBinding)

  TransformBinding() = default;
  TransformBinding(scene::LayerNodeRef target);
  TransformBinding(scene::LayerNodeRef target,
      const float *timeBase,
      const math::float4 *rotation,
      const math::float3 *translation,
      const math::float3 *scale,
      size_t count);

  scene::LayerNodeRef target() const;
  const std::vector<float> &timeBase() const;
  const std::vector<math::float4> &rotation() const;
  const std::vector<math::float3> &translation() const;
  const std::vector<math::float3> &scale() const;

  size_t sampleCount() const;
  float timeBase(size_t i) const;
  const math::float4 &rotation(size_t i) const;
  const math::float3 &translation(size_t i) const;
  const math::float3 &scale(size_t i) const;

  void insertKeyframe(float time, const math::mat4 &m);
  void removeKeyframe(size_t i);

  // Serialization //

  void toDataNode(core::DataNode &node) const;
  void fromDataNode(core::DataNode &node);

 private:
  scene::LayerNodeRef m_target;
  std::vector<float> m_timeBase;
  std::vector<tsd::core::math::float4> m_rotation;
  std::vector<tsd::core::math::float3> m_translation;
  std::vector<tsd::core::math::float3> m_scale;
};

} // namespace tsd::animation
