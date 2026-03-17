// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/Animation.hpp"

namespace tsd::animation {

// Animation //////////////////////////////////////////////////////////////////

ObjectParameterBinding &Animation::addObjectParameterBinding(
    scene::Object *target,
    core::Token paramName,
    ANARIDataType dataType,
    const void *data,
    const float *timeBase,
    size_t count,
    InterpolationRule interp)
{
  ObjectParameterBinding b;
  if (target)
    b.target = *target;
  b.paramName = paramName;
  b.dataType = dataType;
  if (data && count > 0) {
    b.data = TimeSamples(dataType, count);
    b.data.setData(data);
  }
  if (timeBase && count > 0)
    b.timeBase.assign(timeBase, timeBase + count);
  b.interp = interp;
  bindings.push_back(std::move(b));
  return bindings.back();
}

ObjectParameterBinding &Animation::addObjectParameterBinding(
    scene::Object *target,
    core::Token paramName,
    ANARIDataType dataType,
    scene::Object *const *objects,
    const float *timeBase,
    size_t count,
    InterpolationRule interp)
{
  ObjectParameterBinding b;
  if (target)
    b.target = *target;
  b.paramName = paramName;
  b.dataType = dataType;
  if (objects && count > 0) {
    std::vector<size_t> indices(count);
    for (size_t i = 0; i < count; i++)
      indices[i] = objects[i] ? objects[i]->index() : TSD_INVALID_INDEX;
    b.data = TimeSamples(dataType, count);
    b.data.setData(indices.data());
  }
  if (timeBase && count > 0)
    b.timeBase.assign(timeBase, timeBase + count);
  b.interp = interp;
  bindings.push_back(std::move(b));
  return bindings.back();
}

TransformBinding &Animation::addTransformBinding(scene::LayerNodeRef target,
    const float *timeBase,
    const tsd::core::math::float4 *rotation,
    const tsd::core::math::float3 *translation,
    const tsd::core::math::float3 *scale,
    size_t count)
{
  TransformBinding tb;
  tb.target = target;
  tb.timeBase.assign(timeBase, timeBase + count);
  tb.rotation.assign(rotation, rotation + count);
  tb.translation.assign(translation, translation + count);
  tb.scale.assign(scale, scale + count);
  transforms.push_back(std::move(tb));
  return transforms.back();
}

} // namespace tsd::animation
