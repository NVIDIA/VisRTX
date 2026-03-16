// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/Animation.hpp"

namespace tsd::animation {

void Animation::addObjectParameterBinding(scene::Object &target,
    core::Token paramName,
    ANARIDataType dataType,
    const void *data,
    const float *timeBase,
    size_t count,
    InterpolationRule interp)
{
  ObjectParameterBinding b;
  b.target = AnimObjectRef(target);
  b.paramName = paramName;
  b.dataType = dataType;
  b.data = TimeSamples(dataType, count);
  b.data.setData(data);
  b.timeBase.assign(timeBase, timeBase + count);
  b.interp = interp;
  bindings.push_back(std::move(b));
}

void Animation::addTransformBinding(scene::LayerNodeRef target,
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
}

} // namespace tsd::animation
