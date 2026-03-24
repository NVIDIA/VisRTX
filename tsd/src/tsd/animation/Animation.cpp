// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"

namespace tsd::animation {

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
  for (auto &b : m_bindings)
    b.update(t);
  for (auto &tb : m_transforms)
    tb.update(t);
  for (auto &cb : m_customBindings)
    cb.update(t);
}

void Animation::updateObjectDefragmentedIndices(const scene::IndexRemapper &cb)
{
  for (auto &b : m_bindings)
    b.onDefragment(cb);
  for (auto &tb : m_transforms)
    tb.onDefragment(cb);
  for (auto &cb_b : m_customBindings)
    cb_b.onDefragment(cb);
  for (auto &fb : m_fileBindings)
    fb->onDefragment(cb);
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

CallbackBinding &Animation::addCallbackBinding(
    CallbackBinding::Callback callback)
{
  m_customBindings.emplace_back(std::move(callback));
  return m_customBindings.back();
}

const std::vector<CallbackBinding> &Animation::callbackBindings() const
{
  return m_customBindings;
}

CallbackBinding *Animation::editableCallbackBinding(size_t i)
{
  return i < m_customBindings.size() ? &m_customBindings[i] : nullptr;
}

void Animation::removeCallbackBinding(size_t i)
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

const std::vector<std::unique_ptr<FileBinding>> &Animation::fileBindings() const
{
  return m_fileBindings;
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

} // namespace tsd::animation
