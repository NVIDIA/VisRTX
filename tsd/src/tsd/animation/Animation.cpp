// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"

namespace tsd::animation {

// Helper functions ///////////////////////////////////////////////////////////

static TimeSample findTimeSample(const std::vector<float> &times, float t)
{
  if (times.size() == 0)
    return {};
  if (times.size() == 1 || t <= times[0])
    return {0, 0, 0.f};
  if (t >= times.back())
    return {times.size() - 1, times.size() - 1, 0.f};

  auto it = std::upper_bound(cbegin(times), cend(times), t);
  auto hi = std::distance(cbegin(times), it);
  auto lo = hi - 1;

  float span = times[hi] - times[lo];
  float alpha = (span > 0.f) ? (t - times[lo]) / span : 0.f;
  return {size_t(lo), size_t(hi), alpha};
}

static math::mat4 composeTransform(
    math::float4 quat, math::float3 trans, math::float3 scl)
{
  float x = quat.x, y = quat.y, z = quat.z, w = quat.w;
  float x2 = x + x, y2 = y + y, z2 = z + z;
  float xx = x * x2, xy = x * y2, xz = x * z2;
  float yy = y * y2, yz = y * z2, zz = z * z2;
  float wx = w * x2, wy = w * y2, wz = w * z2;

  math::mat4 m;
  m[0] = {(1.f - (yy + zz)) * scl.x, (xy + wz) * scl.x, (xz - wy) * scl.x, 0.f};
  m[1] = {(xy - wz) * scl.y, (1.f - (xx + zz)) * scl.y, (yz + wx) * scl.y, 0.f};
  m[2] = {(xz + wy) * scl.z, (yz - wx) * scl.z, (1.f - (xx + yy)) * scl.z, 0.f};
  m[3] = {trans.x, trans.y, trans.z, 1.f};
  return m;
}

// Animation definitions //////////////////////////////////////////////////////

Animation::Animation(AnimationManager *manager, const std::string &name)
    : m_manager(manager), m_name(name)
{}

Animation::~Animation() = default;

AnimationManager *Animation::manager() const
{
  return m_manager;
}

const std::string &Animation::name() const
{
  return m_name;
}

std::string &Animation::editableName()
{
  return m_name;
}

void Animation::setAnimationTime(float t)
{
  auto r = evaluate(t);
  applyResults(r, *m_manager->scene());
  for (const auto &cb : m_customBindings)
    cb.invoke(t);
}

void Animation::updateObjectDefragmentedIndices(const scene::IndexRemapper &cb)
{
  for (auto &b : m_bindings)
    b.updateObjectDefragmentedIndices(cb);
}

ObjectParameterBinding &Animation::addObjectParameterBinding(
    scene::Object *target,
    core::Token paramName,
    anari::DataType type,
    const void *data,
    const float *timeBase,
    size_t count,
    InterpolationRule interp)
{
  m_bindings.emplace_back(
      target, paramName, type, data, timeBase, count, interp);
  return m_bindings.back();
}

ObjectParameterBinding &Animation::addObjectParameterBinding(
    scene::Object *target,
    core::Token paramName,
    anari::DataType type,
    scene::Object *const *objects,
    const float *timeBase,
    size_t count,
    InterpolationRule interp)
{
  m_bindings.emplace_back(
      target, paramName, type, objects, timeBase, count, interp);
  return m_bindings.back();
}

const std::vector<ObjectParameterBinding> &Animation::objectParameterBindings()
    const
{
  return m_bindings;
}

ObjectParameterBinding *Animation::editableObjectParameterBinding(size_t i)
{
  return i < m_bindings.size() ? &m_bindings[i] : nullptr;
}

void Animation::removeObjectParameterBinding(size_t i)
{
  if (i < m_bindings.size())
    m_bindings.erase(m_bindings.begin() + i);
}

ObjectCustomBinding &Animation::addCustomObjectBinding(
    scene::Object *target, ObjectCustomBinding::Callback callback)
{
  m_customBindings.emplace_back(target, std::move(callback));
  return m_customBindings.back();
}

const std::vector<ObjectCustomBinding> &Animation::customObjectBindings() const
{
  return m_customBindings;
}

ObjectCustomBinding *Animation::editableCustomObjectBinding(size_t i)
{
  return i < m_customBindings.size() ? &m_customBindings[i] : nullptr;
}

void Animation::removeCustomObjectBinding(size_t i)
{
  if (i < m_customBindings.size())
    m_customBindings.erase(m_customBindings.begin() + i);
}

TransformBinding &Animation::addTransformBinding(scene::LayerNodeRef target)
{
  m_transforms.emplace_back(target);
  return m_transforms.back();
}

TransformBinding &Animation::addTransformBinding(scene::LayerNodeRef target,
    const float *timeBase,
    const tsd::core::math::float4 *rotation,
    const tsd::core::math::float3 *translation,
    const tsd::core::math::float3 *scale,
    size_t count)
{
  m_transforms.emplace_back(
      target, timeBase, rotation, translation, scale, count);
  return m_transforms.back();
}

const std::vector<TransformBinding> &Animation::transformBindings() const
{
  return m_transforms;
}

TransformBinding *Animation::editableTransformBinding(size_t i)
{
  return i < m_transforms.size() ? &m_transforms[i] : nullptr;
}

void Animation::removeTransformBinding(size_t i)
{
  if (i < m_transforms.size())
    m_transforms.erase(m_transforms.begin() + i);
}

void Animation::toDataNode(core::DataNode &node) const
{
  auto &scene = *m_manager->scene();

  node["name"] = name();

  auto &bindingsNode = node["objectBindings"];
  for (const auto &b : objectParameterBindings())
    b.toDataNode(bindingsNode.append());

  auto &transformsNode = node["transformBindings"];
  for (const auto &tb : transformBindings())
    tb.toDataNode(transformsNode.append());
}

void Animation::fromDataNode(core::DataNode &node)
{
  m_bindings.clear();
  m_transforms.clear();

  m_name = node["name"].getValueAs<std::string>();

  if (auto *bindingsNode = node.child("objectBindings")) {
    bindingsNode->foreach_child([&](core::DataNode &bn) {
      auto &b = addEmptyObjectParameterBinding();
      b.fromDataNode(bn);
    });
  }

  if (auto *transformsNode = node.child("transformBindings")) {
    transformsNode->foreach_child([&](core::DataNode &tn) {
      auto &b = addEmptyTransformBinding();
      b.fromDataNode(tn);
    });
  }
}

ObjectParameterBinding &Animation::addEmptyObjectParameterBinding()
{
  m_bindings.emplace_back(m_manager->scene());
  return m_bindings.back();
}

TransformBinding &Animation::addEmptyTransformBinding()
{
  m_transforms.emplace_back(m_manager->scene());
  return m_transforms.back();
}

core::Any Animation::interpolateBinding(
    const ObjectParameterBinding &binding, const TimeSample &sample) const
{
  const auto *data = static_cast<const uint8_t *>(binding.data().data());
  if (!data)
    return {};

  size_t elemSize = anari::sizeOf(binding.type());
  if (elemSize == 0)
    return {};

  if (binding.interpolation() == InterpolationRule::STEP
      || sample.lo == sample.hi) {
    size_t idx = (sample.alpha < 0.5f) ? sample.lo : sample.hi;
    return core::Any(binding.type(), data + idx * elemSize);
  }

  if (binding.interpolation() == InterpolationRule::LINEAR) {
    switch (binding.type()) {
    case ANARI_FLOAT32: {
      auto *arr = reinterpret_cast<const float *>(data);
      float v = tsd::math::lerp(arr[sample.lo], arr[sample.hi], sample.alpha);
      return core::Any(v);
    }
    case ANARI_FLOAT32_VEC2: {
      auto *arr = reinterpret_cast<const math::float2 *>(data);
      auto v = tsd::math::lerp(arr[sample.lo], arr[sample.hi], sample.alpha);
      return core::Any(v);
    }
    case ANARI_FLOAT32_VEC3: {
      auto *arr = reinterpret_cast<const math::float3 *>(data);
      auto v = tsd::math::lerp(arr[sample.lo], arr[sample.hi], sample.alpha);
      return core::Any(v);
    }
    case ANARI_FLOAT32_VEC4: {
      auto *arr = reinterpret_cast<const math::float4 *>(data);
      auto v = tsd::math::lerp(arr[sample.lo], arr[sample.hi], sample.alpha);
      return core::Any(v);
    }
    default:
      break;
    }
  }

  // Unsupported interpolation/type combination — fall back to STEP
  size_t idx = (sample.alpha < 0.5f) ? sample.lo : sample.hi;
  return core::Any(binding.type(), data + idx * elemSize);
}

Animation::EvaluationResult Animation::evaluate(float currentTime)
{
  EvaluationResult result;

  for (const auto &binding : m_bindings) {
    if (binding.timeBase().empty())
      continue;

    auto sample = findTimeSample(binding.timeBase(), currentTime);
    auto value = interpolateBinding(binding, sample);
    if (value.valid()) {
      result.parameters.push_back(
          {binding.target(), binding.paramName(), std::move(value)});
    }
  }

  for (const auto &tb : m_transforms) {
    if (tb.timeBase().empty())
      continue;

    auto sample = findTimeSample(tb.timeBase(), currentTime);

    math::float4 rot;
    math::float3 t, s;

    if (sample.lo == sample.hi) {
      rot = tb.rotation(sample.lo);
      t = tb.translation(sample.lo);
      s = tb.scale(sample.lo);
    } else {
      rot = tsd::math::qslerp(
          tb.rotation(sample.lo), tb.rotation(sample.hi), sample.alpha);
      t = tsd::math::slerp(
          tb.translation(sample.lo), tb.translation(sample.hi), sample.alpha);
      s = tsd::math::slerp(
          tb.scale(sample.lo), tb.scale(sample.hi), sample.alpha);
    }

    result.transforms.push_back({tb.target(), composeTransform(rot, t, s)});
  }

  return result;
}

void Animation::applyResults(EvaluationResult &result, scene::Scene &scene)
{
  for (auto &sub : result.parameters) {
    if (!sub.target)
      continue;
    if (anari::isObject(sub.value.type())) {
      auto *obj = scene.getObject(sub.value);
      if (obj)
        sub.target->setParameterObject(sub.paramName, *obj);
    } else {
      sub.target->setParameter(
          sub.paramName, sub.value.type(), sub.value.data());
    }
  }

  for (auto &sub : result.transforms) {
    if (!sub.target)
      continue;
    (*sub.target)->setAsTransform(sub.transform);
    scene.signalLayerTransformChanged(sub.target->value().layer());
  }
}

} // namespace tsd::animation
