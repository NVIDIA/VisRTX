// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Any.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/scene/AnyObjectUsePtr.hpp"
#include "tsd/core/scene/ObjectUsePtr.hpp"
#include "tsd/core/scene/objects/Array.hpp"
// std
#include <string>
#include <vector>

namespace tsd::core {

struct Scene;
using TimeStepArrays = std::vector<ObjectUsePtr<Array>>;
using TimeStepValues = std::vector<Any>;

struct Animation
{
  ~Animation() = default;

  Scene *scene() const;

  std::string &name();
  const std::string &name() const;
  const std::string &info() const;

  void setAsTimeSteps(Object &obj, Token parameter, const TimeStepValues &steps);

  void setAsTimeSteps(
      Object &obj, Token parameter, const TimeStepArrays &steps);

  void setAsTimeSteps(Object &obj,
      const std::vector<Token> &parameters,
      const std::vector<TimeStepValues> &steps);

  void setAsTimeSteps(Object &obj,
      const std::vector<Token> &parameters,
      const std::vector<TimeStepArrays> &steps);

  void update(float time);

  void serialize(DataNode &node) const;
  void deserialize(DataNode &node);

 private:
  Animation(Scene *s, const char *name);

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
    size_t currentStep{TSD_INVALID_INDEX};
  } m_timesteps;
};

} // namespace tsd::core