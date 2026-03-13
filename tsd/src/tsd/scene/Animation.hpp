// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Any.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/scene/AnyObjectUsePtr.hpp"
#include "tsd/scene/Layer.hpp"
#include "tsd/scene/ObjectUsePtr.hpp"
#include "tsd/scene/objects/Array.hpp"
// std
#include <string>
#include <vector>

namespace tsd::core {

struct Scene;
using TimeStepValues = ObjectUsePtr<Array>;
using TimeStepArrays = std::vector<TimeStepValues>;

// Keyframe types ///////////////////////////////////////////////////////////////

struct TransformKeyframe
{
  float time{0.f};      // normalized [0.0, 1.0]
  math::mat4 matrix;    // transform at this keyframe
};

struct ValueKeyframe
{
  float time{0.f};
  Any value;
};

struct KeyframeChannel
{
  Token parameterName;
  ANARIDataType type{ANARI_UNKNOWN};
  std::vector<ValueKeyframe> keyframes; // kept sorted by time
};

///////////////////////////////////////////////////////////////////////////////

struct Animation
{
  ~Animation() = default;

  Scene *scene() const;

  std::string &name();
  const std::string &name() const;
  const std::string &info() const;

  void setAsTimeSteps(
      Object &obj, Token parameter, const TimeStepValues &steps);

  void setAsTimeSteps(
      Object &obj, Token parameter, const TimeStepArrays &steps);

  void setAsTimeSteps(Object &obj,
      const std::vector<Token> &parameters,
      const std::vector<TimeStepValues> &steps);

  void setAsTimeSteps(Object &obj,
      const std::vector<Token> &parameters,
      const std::vector<TimeStepArrays> &steps);

  void setAsTransformSteps(LayerNodeRef node, std::vector<math::mat4> frames);

  // Keyframe API //

  void setKeyframeTargetNode(LayerNodeRef node);
  void setKeyframeTargetObject(Object &obj);

  void addTransformKeyframe(float time, math::mat4 mat);
  bool removeTransformKeyframe(size_t index);

  void addValueKeyframe(
      Token param, ANARIDataType type, float time, const void *value);
  bool removeValueKeyframe(Token param, size_t index);

  bool hasKeyframes() const;
  const std::vector<TransformKeyframe> &transformKeyframes() const;
  const std::vector<KeyframeChannel> &keyframeChannels() const;
  LayerNodeRef keyframeTargetNode() const;
  const Object *keyframeTargetObject() const;

  void update(float time);

  bool targetsObject(const Object *obj) const;
  size_t timeStepCount() const;

  void serialize(DataNode &node) const;
  void deserialize(DataNode &node);

 private:
  Animation(Scene *s, const char *name);
  void updateInfoString(float time, bool cellCentered);

  static Any interpolateAny(
      const Any &a, const Any &b, float alpha, ANARIDataType type);

  friend struct Scene;

  Scene *m_scene{nullptr};
  std::string m_name;
  std::string m_info{"<incomplete animation>"};

  struct TimeStepData
  {
    AnyObjectUsePtr object;
    std::vector<Token> parameterName;
    std::vector<TimeStepArrays> stepsArrays;
    std::vector<TimeStepValues> stepsValues;

    LayerNodeRef transformNode;
    std::vector<math::mat4> transformFrames;
  } m_timesteps;

  struct KeyframeData
  {
    AnyObjectUsePtr object;
    std::vector<KeyframeChannel> channels;
    LayerNodeRef transformNode;
    std::vector<TransformKeyframe> transformKeyframes; // sorted by time
  } m_keyframes;
};

} // namespace tsd::core