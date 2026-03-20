// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Binding.hpp"
// tsd_core
#include "tsd/core/AnyArray.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/TypeMacros.hpp"
// tsd_animation
#include "tsd/animation/Interpolation.hpp"
// tsd_scene
#include "tsd/scene/AnyObjectUsePtr.hpp"
#include "tsd/scene/DefragCallback.hpp"
// std
#include <stdexcept>
#include <vector>

namespace tsd::animation {

/*
 * Binding that drives a single named parameter on a scene object over time,
 * interpolating typed keyframe values stored in an AnyArray buffer.
 *
 * Example:
 *   ObjectParameterBinding b(obj, "opacity", ANARI_FLOAT32,
 *       data, times, count, InterpolationRule::LINEAR);
 *   float v = b.data().dataAs<float>()[0];
 */
struct ObjectParameterBinding : public Binding
{
  TSD_DEFAULT_COPYABLE(ObjectParameterBinding)
  TSD_DEFAULT_MOVEABLE(ObjectParameterBinding)

  ObjectParameterBinding() = default;
  ObjectParameterBinding(scene::Object *target,
      core::Token paramName,
      anari::DataType type,
      const void *data,
      const float *timeBase,
      size_t count,
      InterpolationRule interp = InterpolationRule::LINEAR);

  scene::Object *target() const;
  core::Token paramName() const;
  anari::DataType type() const;
  const core::AnyArray &data() const;
  const std::vector<float> &timeBase() const;
  InterpolationRule interpolation() const;

  template <typename T>
  void insertKeyframe(float time, const T &value);
  void removeKeyframe(size_t i);

  void updateObjectDefragmentedIndices(const scene::IndexRemapper &cb);

  // Serialization //

  void toDataNode(core::DataNode &node) const;
  void fromDataNode(core::DataNode &node);

 private:
  void insertKeyframeImpl(float time, const void *value);

  scene::AnyObjectUsePtr<scene::Object::UseKind::ANIM> m_target;
  core::Token m_paramName;
  anari::DataType m_type{ANARI_UNKNOWN};
  core::AnyArray m_data;
  std::vector<float> m_timeBase;
  InterpolationRule m_interp{InterpolationRule::STEP};
  std::vector<scene::AnyObjectUsePtr<scene::Object::UseKind::ANIM>>
      m_objectRefs;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline void ObjectParameterBinding::insertKeyframe(float time, const T &value)
{
  if (anari::ANARITypeFor<T>::value != type()) {
    throw std::runtime_error(
        "ObjectParameterBinding::insertKeyframe<T>()"
        " called with mismatched value type");
  }

  insertKeyframeImpl(time, &value);
}

} // namespace tsd::animation
