// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/ObjectCustomBinding.hpp"
#include "tsd/animation/ObjectParameterBinding.hpp"
#include "tsd/animation/TransformBinding.hpp"
// tsd_scene
#include "tsd/scene/AnyObjectUsePtr.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <string>
#include <vector>

namespace tsd::animation {

struct AnimationManager;

/*
 * Named animation that owns a collection of parameter and transform bindings,
 * evaluating and applying interpolated values to scene objects at a given time.
 *
 * Example:
 *   Animation &anim = manager.addAnimation("spin");
 *   anim.addTransformBinding(nodeRef, times, rot, trans, scale, count);
 *   anim.setAnimationTime(0.5f);
 */
struct Animation
{
  TSD_DEFAULT_COPYABLE(Animation)
  TSD_DEFAULT_MOVEABLE(Animation)

  Animation(AnimationManager *manager, const std::string &name);
  ~Animation();

  AnimationManager *manager() const;

  const std::string &name() const;
  std::string &editableName();

  void setAnimationTime(float t);
  void updateObjectDefragmentedIndices(const scene::IndexRemapper &cb);

  // Object Parameter Bindings //

  ObjectParameterBinding &addObjectParameterBinding(scene::Object *target,
      core::Token paramName,
      anari::DataType dataType,
      const void *data,
      const float *timeBase,
      size_t count,
      InterpolationRule interp = InterpolationRule::LINEAR);

  ObjectParameterBinding &addObjectParameterBinding(scene::Object *target,
      core::Token paramName,
      anari::DataType dataType,
      scene::Object *const *objects,
      const float *timeBase,
      size_t count,
      InterpolationRule interp = InterpolationRule::STEP);

  const std::vector<ObjectParameterBinding> &objectParameterBindings() const;
  ObjectParameterBinding *editableObjectParameterBinding(size_t i);
  void removeObjectParameterBinding(size_t i);

  // Custom Object Bindings //

  ObjectCustomBinding &addCustomObjectBinding(
      scene::Object *target, ObjectCustomBinding::Callback callback);

  const std::vector<ObjectCustomBinding> &customObjectBindings() const;
  ObjectCustomBinding *editableCustomObjectBinding(size_t i);
  void removeCustomObjectBinding(size_t i);

  // Transform Bindings //

  TransformBinding &addTransformBinding(scene::LayerNodeRef target);
  TransformBinding &addTransformBinding(scene::LayerNodeRef target,
      const float *timeBase,
      const tsd::core::math::float4 *rotation,
      const tsd::core::math::float3 *translation,
      const tsd::core::math::float3 *scale,
      size_t count);

  const std::vector<TransformBinding> &transformBindings() const;
  TransformBinding *editableTransformBinding(size_t i);
  void removeTransformBinding(size_t i);

  // Serialization //

  void toDataNode(core::DataNode &node) const;
  void fromDataNode(core::DataNode &node);

 private:
  ObjectParameterBinding &addEmptyObjectParameterBinding();
  TransformBinding &addEmptyTransformBinding();

  struct ParameterSubstitution
  {
    scene::Object *target{nullptr};
    core::Token paramName;
    core::Any value;
  };

  struct TransformSubstitution
  {
    scene::LayerNodeRef target;
    tsd::core::math::mat4 transform;
  };

  struct EvaluationResult
  {
    std::vector<ParameterSubstitution> parameters;
    std::vector<TransformSubstitution> transforms;
  };

  core::Any interpolateBinding(
      const ObjectParameterBinding &binding, const TimeSample &sample) const;

  EvaluationResult evaluate(float currentTime);

  void applyResults(EvaluationResult &result, scene::Scene &scene);

  // Data //

  AnimationManager *m_manager{nullptr};
  std::string m_name{"<unnamed_animation>"};
  std::vector<ObjectParameterBinding> m_bindings;
  std::vector<ObjectCustomBinding> m_customBindings;
  std::vector<TransformBinding> m_transforms;
};

} // namespace tsd::animation
