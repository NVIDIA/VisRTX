// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/TimeSamples.hpp"

// tsd
#include "tsd/core/Any.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/scene/AnyObjectUsePtr.hpp"
#include "tsd/scene/Layer.hpp"
#include "tsd/scene/Object.hpp"

// std
#include <string>
#include <vector>

namespace tsd::animation {

// Interpolation //////////////////////////////////////////////////////////////

enum class InterpolationRule
{
  STEP,
  LINEAR,
  SLERP,
};

// Bindings ///////////////////////////////////////////////////////////////////

struct ObjectParameterBinding
{
  scene::AnyObjectUsePtr<scene::Object::UseKind::ANIM> target;
  core::Token paramName;
  ANARIDataType dataType{ANARI_UNKNOWN};
  TimeSamples data;
  std::vector<float> timeBase;
  InterpolationRule interp{InterpolationRule::STEP};
};

struct TransformBinding
{
  scene::LayerNodeRef target;
  std::vector<float> timeBase;
  std::vector<tsd::core::math::float4> rotation;
  std::vector<tsd::core::math::float3> translation;
  std::vector<tsd::core::math::float3> scale;
};

struct Animation
{
  std::string name;
  std::vector<ObjectParameterBinding> bindings;
  std::vector<TransformBinding> transforms;

  ObjectParameterBinding &addObjectParameterBinding(scene::Object *target,
      core::Token paramName,
      ANARIDataType dataType,
      const void *data,
      const float *timeBase,
      size_t count,
      InterpolationRule interp = InterpolationRule::LINEAR);

  ObjectParameterBinding &addObjectParameterBinding(scene::Object *target,
      core::Token paramName,
      ANARIDataType dataType,
      scene::Object *const *objects,
      const float *timeBase,
      size_t count,
      InterpolationRule interp = InterpolationRule::STEP);

  TransformBinding &addTransformBinding(scene::LayerNodeRef target,
      const float *timeBase,
      const tsd::core::math::float4 *rotation,
      const tsd::core::math::float3 *translation,
      const tsd::core::math::float3 *scale,
      size_t count);
};

// Evaluation /////////////////////////////////////////////////////////////////

struct ParameterSubstitution
{
  scene::AnyObjectUsePtr<scene::Object::UseKind::ANIM> target;
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

EvaluationResult evaluate(
    const std::vector<Animation> &animations, float currentTime);

} // namespace tsd::animation

namespace tsd::scene {
struct Scene;
} // namespace tsd::scene

namespace tsd::animation {
void applyResults(EvaluationResult &result, tsd::scene::Scene &scene);
} // namespace tsd::animation
