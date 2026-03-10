// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/TimeSamples.hpp"
#include "tsd/scene/Layer.hpp"
#include "tsd/scene/Object.hpp"
// tsd
#include <tsd/core/Any.hpp>
#include <tsd/core/TSDMath.hpp>
#include <tsd/core/Token.hpp>
// std
#include <string>
#include <vector>

namespace tsd::animation {

using Token = tsd::core::Token;
using Any = tsd::core::Any;
using Object = tsd::scene::Object *;
using LayerNodeRef = tsd::scene::LayerNodeRef;

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
  Object target;
  Token paramName;
  ANARIDataType dataType{ANARI_UNKNOWN};

  TimeSamples data;
  std::vector<float> timeBase;
  InterpolationRule interp{InterpolationRule::STEP};
};

struct TransformBinding
{
  LayerNodeRef target;
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
};

// Evaluation /////////////////////////////////////////////////////////////////

struct ParameterSubstitution
{
  Object target;
  Token paramName;
  Any value;
};

struct TransformSubstitution
{
  LayerNodeRef target;
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
