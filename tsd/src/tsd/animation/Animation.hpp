// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/CallbackBinding.hpp"
#include "tsd/animation/FileBinding.hpp"
#include "tsd/animation/ObjectParameterBinding.hpp"
#include "tsd/animation/TransformBinding.hpp"
// tsd_scene
#include "tsd/scene/AnyObjectUsePtr.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <memory>
#include <string>
#include <type_traits>
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
  TSD_DEFAULT_MOVEABLE(Animation)

  Animation(AnimationManager *manager, const std::string &name);
  ~Animation();

  AnimationManager *manager() const;

  const std::string &name() const;
  std::string &editableName();

  void setAnimationTime(float t);
  void updateObjectDefragmentedIndices(const scene::IndexRemapper &cb);

  //// Object Parameter Bindings ////

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

  //// Callback Bindings ////

  CallbackBinding &addCallbackBinding(CallbackBinding::Callback callback);

  const std::vector<CallbackBinding> &callbackBindings() const;
  CallbackBinding *editableCallbackBinding(size_t i);
  void removeCallbackBinding(size_t i);

  //// Transform Bindings ////

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

  //// File Bindings ////

  // File bindings are type-erased behind FileBinding* and owned exclusively by
  // the Animation (unique_ptr).  A template emplacement function is used
  // instead of a typed add() so that the concrete type is constructed in-place
  // without requiring the caller to manage its lifetime (e.g. via shared_ptr).
  template <typename T, typename... Args>
  T &emplaceFileBinding(Args &&...args);

  const std::vector<std::unique_ptr<FileBinding>> &fileBindings() const;

  // Deserialization helpers: add an empty binding and return a reference so
  // the caller can populate it via fromDataNode().
  ObjectParameterBinding &addEmptyObjectParameterBinding();
  TransformBinding &addEmptyTransformBinding();

 private:
  // Data //

  AnimationManager *m_manager{nullptr};
  std::string m_name{"<unnamed_animation>"};
  std::vector<ObjectParameterBinding> m_bindings;
  std::vector<CallbackBinding> m_customBindings;
  std::vector<TransformBinding> m_transforms;
  std::vector<std::unique_ptr<FileBinding>> m_fileBindings;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T, typename... Args>
inline T &Animation::emplaceFileBinding(Args &&...args)
{
  static_assert(
      std::is_base_of_v<FileBinding, T>, "T must be derived from FileBinding");
  auto &ptr = m_fileBindings.emplace_back(
      std::make_unique<T>(std::forward<Args>(args)...));
  return static_cast<T &>(*ptr);
}

} // namespace tsd::animation
