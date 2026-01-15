// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/Animation.hpp"

// tsd_core
#include "tsd/core/Logging.hpp"
#include "tsd/core/scene/Scene.hpp"
// std
#include <algorithm>

namespace tsd::core {

// Helper functions ///////////////////////////////////////////////////////////

static size_t calculateIndexForTime(
    float time, size_t numSteps, bool cellCentered)
{
  time = std::clamp(time, 0.f, 1.f);
  // TODO: centeredness should be used when value interpolation is implemented
  return static_cast<size_t>(std::round(time * (numSteps - 1)));
}

// Animation definitions //////////////////////////////////////////////////////

Animation::Animation(Scene *s, const char *name) : m_scene(s), m_name(name) {}

Scene *Animation::scene() const
{
  return m_scene;
}

std::string &Animation::name()
{
  return m_name;
}

const std::string &Animation::name() const
{
  return m_name;
}

const std::string &Animation::info() const
{
  return m_info;
}

void Animation::setAsTimeSteps(
    Object &obj, Token parameter, const TimeStepValues &steps)
{
  setAsTimeSteps(
      obj, std::vector<Token>{parameter}, std::vector<TimeStepValues>{steps});
}

void Animation::setAsTimeSteps(
    Object &obj, Token parameter, const TimeStepArrays &steps)
{
  setAsTimeSteps(
      obj, std::vector<Token>{parameter}, std::vector<TimeStepArrays>{steps});
}

void Animation::setAsTimeSteps(Object &obj,
    const std::vector<Token> &parameters,
    const std::vector<TimeStepValues> &steps)
{
  if (parameters.size() != steps.size()) {
    logError(
        "[AnimatedTimeSeries::setAsTimeSteps()] parameter/steps size mismatch");
    return;
  }

  m_timesteps.object = obj;
  m_timesteps.parameterName = parameters;
  m_timesteps.stepsValues = steps;
  m_timesteps.stepsArrays.clear();
  updateInfoString(0.f, false);
}

void Animation::setAsTimeSteps(Object &obj,
    const std::vector<Token> &parameters,
    const std::vector<TimeStepArrays> &steps)
{
  if (parameters.size() != steps.size()) {
    logError(
        "[AnimatedTimeSeries::setAsTimeSteps()] parameter/steps size mismatch");
    return;
  }

  m_timesteps.object = obj;
  m_timesteps.parameterName = parameters;
  m_timesteps.stepsValues.clear();
  m_timesteps.stepsArrays = steps;
  updateInfoString(0.f, true);
}

void Animation::update(float time)
{
  auto &ts = m_timesteps;

  if ((ts.stepsValues.empty() && ts.stepsArrays.empty()) || !ts.object
      || ts.parameterName.empty()) {
    logWarning(
        "[AnimatedTimeSeries::update()] incomplete animation object '%s'",
        name().c_str());
    return;
  }

  if (!ts.stepsValues.empty()) {
    // TODO(jda): (linearly) interpolate between time steps for values?
    for (size_t i = 0; i < ts.stepsValues.size(); i++) {
      const auto &a = *ts.stepsValues[i];
      const size_t idx = calculateIndexForTime(time, a.size(), false);
      ts.object->setParameter(
          ts.parameterName[i], a.elementType(), a.elementAt(idx));
    }
    updateInfoString(time, false);
  } else if (!ts.stepsArrays.empty()) {
    for (size_t i = 0; i < ts.stepsArrays.size(); i++) {
      const auto &c = ts.stepsArrays[i];
      const size_t idx = calculateIndexForTime(time, c.size(), true);
      ts.object->setParameterObject(ts.parameterName[i], *c[idx]);
    }
    updateInfoString(time, true);
  }
}

bool Animation::targetsObject(const Object *obj) const
{
  return obj && m_timesteps.object && m_timesteps.object.get() == obj;
}

size_t Animation::timeStepCount() const
{
  const auto &ts = m_timesteps;
  if (!ts.stepsValues.empty()) {
    const auto &steps = ts.stepsValues.front();
    return steps ? steps->size() : 0;
  }
  if (!ts.stepsArrays.empty())
    return ts.stepsArrays.front().size();
  return 0;
}

void Animation::serialize(DataNode &node) const
{
  node["name"] = name();

  // Serialize as timestep animation //

  auto &ts = m_timesteps;

  // Write node values //

  auto &timeseries = node["timeseries"];
  timeseries["object"] = ts.object
      ? tsd::core::Any(ts.object->type(), ts.object->index())
      : tsd::core::Any();
  timeseries["kind"] = ts.stepsArrays.empty() ? "values" : "arrays";

  // Write animation sets //

  auto &animationSets = timeseries["animationSets"];
  if (!ts.stepsArrays.empty()) {
    for (size_t i = 0; i < ts.stepsArrays.size(); i++) {
      auto &setNode = animationSets.append();
      setNode["parameterName"] = ts.parameterName[i].str();

      std::vector<size_t> setArrayIndices;
      setArrayIndices.reserve(ts.stepsArrays[i].size());

      for (auto &s : ts.stepsArrays[i])
        setArrayIndices.push_back(s->index());
      setNode["steps"].setValueAsArray(setArrayIndices);
    }
  } else if (!ts.stepsValues.empty()) {
    for (size_t i = 0; i < ts.stepsValues.size(); i++) {
      auto &setNode = animationSets.append();
      setNode["parameterName"] = ts.parameterName[i].str();
      setNode["steps"].setValue(ts.stepsValues[i]->index());
    }
  }
}

void Animation::deserialize(DataNode &node)
{
  auto getTimeStepArrays = [this](DataNode &setNode) {
    size_t numSteps = 0;
    void *data = nullptr;
    ANARIDataType type = ANARI_UNKNOWN;

    auto &stepsNode = setNode["steps"];
    stepsNode.getValueAsArray(&type, &data, &numSteps);

    TimeStepArrays stepsArrays;

    if (type == anari::ANARITypeFor<size_t>::value) {
      auto indices = static_cast<size_t *>(data);
      for (size_t i = 0; i < numSteps; i++)
        stepsArrays.emplace_back(m_scene->getObject<Array>(indices[i]));
    } else {
      logError(
          "[Animation::deserialize()] invalid data type for timestep arrays");
    }

    return stepsArrays;
  };

  //////////////////

  name() = node["name"].getValueAs<std::string>();

  if (auto *c = node.child("timeseries"); c != nullptr) {
    auto &ts = m_timesteps;
    auto &tsNode = *c;

    auto *kindNode = node.child("timeseries")->child("kind");
    bool isArrayBased =
        !kindNode || kindNode->getValueAs<std::string>() == "arrays";

    auto object = m_scene->getObject(tsNode["object"].getValue());

    if (auto *sets = tsNode.child("animationSets"); sets != nullptr) {
      std::vector<Token> parameterNames;
      std::vector<TimeStepArrays> allSteps;
      std::vector<TimeStepValues> allValueSteps;

      sets->foreach_child([&](DataNode &setNode) {
        auto parameterName =
            Token(setNode["parameterName"].getValueAs<std::string>().c_str());
        parameterNames.push_back(parameterName);
        if (isArrayBased)
          allSteps.push_back(getTimeStepArrays(setNode));
        else {
          size_t stepIdx = setNode["steps"].getValueAs<size_t>();
          auto arr = m_scene->getObject<Array>(stepIdx);
          if (arr)
            allValueSteps.push_back(arr);
          else {
            logError(
                "[Animation::deserialize()] invalid array index %zu", stepIdx);
          }
        }
      });

      if (isArrayBased)
        setAsTimeSteps(*object, parameterNames, allSteps);
      else
        setAsTimeSteps(*object, parameterNames, allValueSteps);
    }
  }
}

void Animation::updateInfoString(float time, bool cellCentered)
{
  auto &ts = m_timesteps;
  auto doUpdate = [&](size_t numSteps) {
    const size_t idx = calculateIndexForTime(time, numSteps, cellCentered);
    m_info = "current timestep: " + std::to_string(idx) + "/"
        + std::to_string(numSteps - 1);
  };

  if (!ts.stepsValues.empty() && ts.stepsValues[0]) {
    const auto &a = *ts.stepsValues[0];
    doUpdate(a.size());
  } else if (!ts.stepsArrays.empty() && !ts.stepsArrays[0].empty()) {
    doUpdate(ts.stepsArrays[0].size());
  } else {
    m_info = "<incomplete animation>";
  }
}

} // namespace tsd::core
