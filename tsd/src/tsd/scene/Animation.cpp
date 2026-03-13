// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/scene/Animation.hpp"

// tsd_core
#include "tsd/core/Logging.hpp"
#include "tsd/scene/Scene.hpp"
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

static float lerpF(float a, float b, float t)
{
  return a + t * (b - a);
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

void Animation::setAsTransformSteps(
    LayerNodeRef node, std::vector<math::mat4> frames)
{
  m_timesteps = {};
  m_timesteps.transformNode = node;
  m_timesteps.transformFrames = std::move(frames);
  updateInfoString(0.f, false);
}

// Keyframe API ///////////////////////////////////////////////////////////////

void Animation::setKeyframeTargetNode(LayerNodeRef node)
{
  m_keyframes.transformNode = node;
}

void Animation::setKeyframeTargetObject(Object &obj)
{
  m_keyframes.object = obj;
}

void Animation::addTransformKeyframe(float time, math::mat4 mat)
{
  time = std::clamp(time, 0.f, 1.f);
  auto &kfs = m_keyframes.transformKeyframes;
  // Replace existing keyframe within tolerance
  for (auto &kf : kfs) {
    if (std::abs(kf.time - time) < 1e-4f) {
      kf.matrix = mat;
      return;
    }
  }
  // Insert sorted
  TransformKeyframe kf{time, mat};
  auto it = std::lower_bound(
      kfs.begin(), kfs.end(), kf, [](const auto &a, const auto &b) {
        return a.time < b.time;
      });
  kfs.insert(it, kf);
}

bool Animation::removeTransformKeyframe(size_t index)
{
  auto &kfs = m_keyframes.transformKeyframes;
  if (index >= kfs.size())
    return false;
  kfs.erase(kfs.begin() + index);
  return true;
}

void Animation::addValueKeyframe(
    Token param, ANARIDataType type, float time, const void *value)
{
  time = std::clamp(time, 0.f, 1.f);
  // Find or create channel
  KeyframeChannel *chan = nullptr;
  for (auto &c : m_keyframes.channels) {
    if (c.parameterName == param) {
      chan = &c;
      break;
    }
  }
  if (!chan) {
    m_keyframes.channels.push_back({param, type, {}});
    chan = &m_keyframes.channels.back();
  }

  ValueKeyframe kf{time, Any(type, value)};
  // Replace existing
  for (auto &existing : chan->keyframes) {
    if (std::abs(existing.time - time) < 1e-4f) {
      existing.value = kf.value;
      return;
    }
  }
  // Insert sorted
  auto it = std::lower_bound(chan->keyframes.begin(),
      chan->keyframes.end(),
      kf,
      [](const auto &a, const auto &b) { return a.time < b.time; });
  chan->keyframes.insert(it, kf);
}

bool Animation::removeValueKeyframe(Token param, size_t index)
{
  for (auto &c : m_keyframes.channels) {
    if (c.parameterName == param) {
      if (index >= c.keyframes.size())
        return false;
      c.keyframes.erase(c.keyframes.begin() + index);
      return true;
    }
  }
  return false;
}

bool Animation::hasKeyframes() const
{
  return !m_keyframes.transformKeyframes.empty()
      || !m_keyframes.channels.empty();
}

const std::vector<TransformKeyframe> &Animation::transformKeyframes() const
{
  return m_keyframes.transformKeyframes;
}

const std::vector<KeyframeChannel> &Animation::keyframeChannels() const
{
  return m_keyframes.channels;
}

LayerNodeRef Animation::keyframeTargetNode() const
{
  return m_keyframes.transformNode;
}

const Object *Animation::keyframeTargetObject() const
{
  return m_keyframes.object.get();
}

Any Animation::interpolateAny(
    const Any &a, const Any &b, float alpha, ANARIDataType type)
{
  switch (type) {
  case ANARI_FLOAT32: {
    float va = a.get<float>();
    float vb = b.get<float>();
    return Any(lerpF(va, vb, alpha));
  }
  case ANARI_FLOAT32_VEC2: {
    auto va = a.get<math::float2>();
    auto vb = b.get<math::float2>();
    math::float2 r{lerpF(va.x, vb.x, alpha), lerpF(va.y, vb.y, alpha)};
    return Any(r);
  }
  case ANARI_FLOAT32_VEC3: {
    auto va = a.get<math::float3>();
    auto vb = b.get<math::float3>();
    math::float3 r{lerpF(va.x, vb.x, alpha),
        lerpF(va.y, vb.y, alpha),
        lerpF(va.z, vb.z, alpha)};
    return Any(r);
  }
  case ANARI_FLOAT32_VEC4: {
    auto va = a.get<math::float4>();
    auto vb = b.get<math::float4>();
    math::float4 r{lerpF(va.x, vb.x, alpha),
        lerpF(va.y, vb.y, alpha),
        lerpF(va.z, vb.z, alpha),
        lerpF(va.w, vb.w, alpha)};
    return Any(r);
  }
  default:
    // Snap to nearest for non-float types
    return alpha < 0.5f ? a : b;
  }
}

void Animation::update(float time)
{
  auto &ts = m_timesteps;

  if (!ts.transformFrames.empty()) {
    if (!ts.transformNode) {
      logWarning(
          "[AnimatedTimeSeries::update()] transform animation '%s' has no node",
          name().c_str());
      return;
    }
    const size_t idx =
        calculateIndexForTime(time, ts.transformFrames.size(), false);
    (*ts.transformNode)->setAsTransform(ts.transformFrames[idx]);
    m_scene->signalLayerTransformChanged((*ts.transformNode)->layer());
    updateInfoString(time, false);
    return;
  }

  if ((ts.stepsValues.empty() && ts.stepsArrays.empty()) || !ts.object
      || ts.parameterName.empty()) {
    if (!hasKeyframes()) {
      logWarning(
          "[AnimatedTimeSeries::update()] incomplete animation object '%s'",
          name().c_str());
    }
  } else if (!ts.stepsValues.empty()) {
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

  // Keyframe animation //

  auto &kf = m_keyframes;

  if (!kf.transformKeyframes.empty() && kf.transformNode) {
    const auto &kfs = kf.transformKeyframes;
    math::mat4 result;
    if (kfs.size() == 1) {
      result = kfs[0].matrix;
    } else if (time <= kfs.front().time) {
      result = kfs.front().matrix;
    } else if (time >= kfs.back().time) {
      result = kfs.back().matrix;
    } else {
      // Binary search for surrounding pair
      auto it = std::lower_bound(kfs.begin(),
          kfs.end(),
          time,
          [](const TransformKeyframe &k, float t) { return k.time < t; });
      const auto &k1 = *it;
      const auto &k0 = *(it - 1);
      float alpha = (time - k0.time) / (k1.time - k0.time);
      // Element-wise mat4 lerp
      for (int col = 0; col < 4; col++) {
        result[col][0] = lerpF(k0.matrix[col][0], k1.matrix[col][0], alpha);
        result[col][1] = lerpF(k0.matrix[col][1], k1.matrix[col][1], alpha);
        result[col][2] = lerpF(k0.matrix[col][2], k1.matrix[col][2], alpha);
        result[col][3] = lerpF(k0.matrix[col][3], k1.matrix[col][3], alpha);
      }
    }
    (*kf.transformNode)->setAsTransform(result);
    m_scene->signalLayerTransformChanged((*kf.transformNode)->layer());
  }

  if (!kf.channels.empty() && kf.object) {
    for (const auto &chan : kf.channels) {
      const auto &cks = chan.keyframes;
      if (cks.empty())
        continue;

      Any result;
      if (cks.size() == 1) {
        result = cks[0].value;
      } else if (time <= cks.front().time) {
        result = cks.front().value;
      } else if (time >= cks.back().time) {
        result = cks.back().value;
      } else {
        auto it = std::lower_bound(
            cks.begin(), cks.end(), time, [](const ValueKeyframe &k, float t) {
              return k.time < t;
            });
        const auto &vk1 = *it;
        const auto &vk0 = *(it - 1);
        float alpha = (time - vk0.time) / (vk1.time - vk0.time);
        result = interpolateAny(vk0.value, vk1.value, alpha, chan.type);
      }

      if (result)
        kf.object->setParameter(chan.parameterName, chan.type, result.data());
    }
  }
}

bool Animation::targetsObject(const Object *obj) const
{
  return obj && m_timesteps.object && m_timesteps.object.get() == obj;
}

size_t Animation::timeStepCount() const
{
  const auto &ts = m_timesteps;
  if (!ts.transformFrames.empty())
    return ts.transformFrames.size();
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

  // Serialize keyframe animation //

  auto &kf = m_keyframes;
  if (!hasKeyframes())
    return;

  auto &kfNode = node["keyframes"];

  // Transform node reference
  if (kf.transformNode) {
    auto *lay = (*kf.transformNode)->layer();
    kfNode["nodeLayer"] = m_scene->getLayerName(lay).str();
    kfNode["nodeIndex"] = kf.transformNode->index();
  }

  // Camera/object reference
  if (kf.object) {
    kfNode["object"] = tsd::core::Any(kf.object->type(), kf.object->index());
  }

  // Transform keyframes: time + flat mat4 (16 floats)
  auto &tkfsNode = kfNode["transformKeyframes"];
  for (const auto &tk : kf.transformKeyframes) {
    auto &tkNode = tkfsNode.append();
    tkNode["time"] = tk.time;
    float m[16];
    for (int c = 0; c < 4; c++)
      for (int r = 0; r < 4; r++)
        m[c * 4 + r] = tk.matrix[c][r];
    tkNode["matrix"].setValueAsArray<float>(m, 16);
  }

  // Value keyframe channels — store values as raw float arrays for safe
  // round-tripping through the DataTree file format
  auto &chansNode = kfNode["channels"];
  for (const auto &chan : kf.channels) {
    auto &chanNode = chansNode.append();
    chanNode["parameterName"] = chan.parameterName.str();
    chanNode["type"] = (int)chan.type;
    size_t numFloats = anari::sizeOf(chan.type) / sizeof(float);
    auto &vkfsNode = chanNode["keyframes"];
    for (const auto &vk : chan.keyframes) {
      auto &vkNode = vkfsNode.append();
      vkNode["time"] = vk.time;
      vkNode["value"].setValueAsArray<float>(
          static_cast<const float *>(vk.value.data()), numFloats);
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

      if (object) {
        if (isArrayBased)
          setAsTimeSteps(*object, parameterNames, allSteps);
        else
          setAsTimeSteps(*object, parameterNames, allValueSteps);
      }
    }
  }

  // Deserialize keyframe animation //

  if (auto *kfData = node.child("keyframes"); kfData != nullptr) {
    auto &kf = m_keyframes;

    // Restore transform node reference
    if (auto *nlNode = kfData->child("nodeLayer")) {
      auto layerName = Token(nlNode->getValueAs<std::string>().c_str());
      auto *lay = m_scene->layer(layerName);
      if (lay) {
        size_t idx = (*kfData)["nodeIndex"].getValueAs<size_t>();
        kf.transformNode = lay->at(idx);
      }
    }

    // Restore camera/object reference
    if (auto *objNode = kfData->child("object")) {
      auto *obj = m_scene->getObject(objNode->getValue());
      if (obj)
        kf.object = *obj; // use operator=(Object&), NOT the constructor (which
                          // is a no-op)
    }

    // Restore transform keyframes
    if (auto *tkfsNode = kfData->child("transformKeyframes")) {
      tkfsNode->foreach_child([&](DataNode &tkNode) {
        TransformKeyframe tk;
        tk.time = tkNode["time"].getValueAs<float>();
        ANARIDataType type = ANARI_UNKNOWN;
        void *data = nullptr;
        size_t numVals = 0;
        tkNode["matrix"].getValueAsArray(&type, &data, &numVals);
        if (type == ANARI_FLOAT32 && numVals == 16) {
          auto *m = static_cast<float *>(data);
          for (int c = 0; c < 4; c++)
            for (int r = 0; r < 4; r++)
              tk.matrix[c][r] = m[c * 4 + r];
        }
        kf.transformKeyframes.push_back(std::move(tk));
      });
    }

    // Restore value keyframe channels
    if (auto *chansNode = kfData->child("channels")) {
      chansNode->foreach_child([&](DataNode &chanNode) {
        KeyframeChannel chan;
        chan.parameterName =
            Token(chanNode["parameterName"].getValueAs<std::string>().c_str());
        chan.type = (ANARIDataType)chanNode["type"].getValueAs<int>();
        if (auto *vkfsNode = chanNode.child("keyframes")) {
          vkfsNode->foreach_child([&](DataNode &vkNode) {
            ValueKeyframe vk;
            vk.time = vkNode["time"].getValueAs<float>();
            ANARIDataType valType = ANARI_UNKNOWN;
            void *valData = nullptr;
            size_t valCount = 0;
            vkNode["value"].getValueAsArray(&valType, &valData, &valCount);
            if (valData && valCount > 0)
              vk.value = Any(chan.type, valData);
            chan.keyframes.push_back(std::move(vk));
          });
        }
        kf.channels.push_back(std::move(chan));
      });
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

  if (!ts.transformFrames.empty()) {
    doUpdate(ts.transformFrames.size());
  } else if (!ts.stepsValues.empty() && ts.stepsValues[0]) {
    const auto &a = *ts.stepsValues[0];
    doUpdate(a.size());
  } else if (!ts.stepsArrays.empty() && !ts.stepsArrays[0].empty()) {
    doUpdate(ts.stepsArrays[0].size());
  } else {
    m_info = "<incomplete animation>";
  }
}

} // namespace tsd::core
