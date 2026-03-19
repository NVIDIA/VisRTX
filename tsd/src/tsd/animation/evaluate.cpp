// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Animation.hpp"
// std
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>

namespace tsd::animation {

using tsd::core::Any;
using namespace tsd::core::math;

// Time base lookup ///////////////////////////////////////////////////////////

struct TimeSample
{
  size_t lo{0};
  size_t hi{0};
  float alpha{0.f};
};

// static TimeSample findTimeSample(const float *times, size_t count, float t)
static TimeSample findTimeSample(const std::vector<float> times, float t)
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

// Scalar interpolation ///////////////////////////////////////////////////////

static float lerpF(float a, float b, float t)
{
  return a + t * (b - a);
}

static float2 lerpV2(float2 a, float2 b, float t)
{
  return {lerpF(a.x, b.x, t), lerpF(a.y, b.y, t)};
}

static float3 lerpV3(float3 a, float3 b, float t)
{
  return {lerpF(a.x, b.x, t), lerpF(a.y, b.y, t), lerpF(a.z, b.z, t)};
}

static float4 lerpV4(float4 a, float4 b, float t)
{
  return {
      lerpF(a.x, b.x, t),
      lerpF(a.y, b.y, t),
      lerpF(a.z, b.z, t),
      lerpF(a.w, b.w, t),
  };
}

// Quaternion SLERP ///////////////////////////////////////////////////////////

static float4 safeNormalize(float4 v)
{
  float len = std::sqrt(tsd::math::dot(v, v));
  return (len > 0.f) ? v / tsd::math::length(v) : v;
}

static float4 slerp(float4 a, float4 b, float t)
{
  float d = tsd::math::dot(a, b);

  // Short path: negate to avoid the long arc
  if (d < 0.f) {
    b = b * -1.f;
    d = -d;
  }

  constexpr float THRESHOLD = 0.9995f;
  if (d > THRESHOLD)
    return safeNormalize(lerpV4(a, b, t));

  float theta = std::acos(std::clamp(d, -1.f, 1.f));
  float sinTheta = std::sin(theta);
  float wa = std::sin((1.f - t) * theta) / sinTheta;
  float wb = std::sin(t * theta) / sinTheta;

  return wa * a + wb * b;
}

// Compose mat4 from decomposed transform /////////////////////////////////////

static mat4 composeTransform(float4 quat, float3 trans, float3 scl)
{
  float x = quat.x, y = quat.y, z = quat.z, w = quat.w;
  float x2 = x + x, y2 = y + y, z2 = z + z;
  float xx = x * x2, xy = x * y2, xz = x * z2;
  float yy = y * y2, yz = y * z2, zz = z * z2;
  float wx = w * x2, wy = w * y2, wz = w * z2;

  mat4 m;
  m[0] = {(1.f - (yy + zz)) * scl.x, (xy + wz) * scl.x, (xz - wy) * scl.x, 0.f};
  m[1] = {(xy - wz) * scl.y, (1.f - (xx + zz)) * scl.y, (yz + wx) * scl.y, 0.f};
  m[2] = {(xz + wy) * scl.z, (yz - wx) * scl.z, (1.f - (xx + yy)) * scl.z, 0.f};
  m[3] = {trans.x, trans.y, trans.z, 1.f};
  return m;
}

// Per-binding interpolation //////////////////////////////////////////////////

static Any interpolateBinding(
    const ObjectParameterBinding &binding, const TimeSample &sample)
{
  const auto *data = static_cast<const uint8_t *>(binding.data.data());
  if (!data)
    return {};

  size_t elemSize = anari::sizeOf(binding.dataType);
  if (elemSize == 0)
    return {};

  if (binding.interp == InterpolationRule::STEP || sample.lo == sample.hi) {
    size_t idx = (sample.alpha < 0.5f) ? sample.lo : sample.hi;
    return Any(binding.dataType, data + idx * elemSize);
  }

  if (binding.interp == InterpolationRule::LINEAR) {
    switch (binding.dataType) {
    case ANARI_FLOAT32: {
      auto *arr = reinterpret_cast<const float *>(data);
      float v = lerpF(arr[sample.lo], arr[sample.hi], sample.alpha);
      return Any(v);
    }
    case ANARI_FLOAT32_VEC2: {
      auto *arr = reinterpret_cast<const float2 *>(data);
      auto v = lerpV2(arr[sample.lo], arr[sample.hi], sample.alpha);
      return Any(v);
    }
    case ANARI_FLOAT32_VEC3: {
      auto *arr = reinterpret_cast<const float3 *>(data);
      auto v = lerpV3(arr[sample.lo], arr[sample.hi], sample.alpha);
      return Any(v);
    }
    case ANARI_FLOAT32_VEC4: {
      auto *arr = reinterpret_cast<const float4 *>(data);
      auto v = lerpV4(arr[sample.lo], arr[sample.hi], sample.alpha);
      return Any(v);
    }
    default:
      break;
    }
  }

  // Unsupported interpolation/type combination — fall back to STEP
  size_t idx = (sample.alpha < 0.5f) ? sample.lo : sample.hi;
  return Any(binding.dataType, data + idx * elemSize);
}

// Evaluate ///////////////////////////////////////////////////////////////////

EvaluationResult evaluate(
    const std::vector<Animation> &animations, float currentTime)
{
  EvaluationResult result;

  for (const auto &anim : animations) {
    for (const auto &binding : anim.bindings) {
      if (binding.timeBase.empty())
        continue;

      auto sample = findTimeSample(binding.timeBase, currentTime);
      auto value = interpolateBinding(binding, sample);
      if (value.valid()) {
        result.parameters.push_back(
            {binding.target, binding.paramName, std::move(value)});
      }
    }

    for (const auto &tb : anim.transforms) {
      if (tb.timeBase.empty())
        continue;

      auto sample = findTimeSample(tb.timeBase, currentTime);

      float4 rot;
      float3 t, s;

      if (sample.lo == sample.hi) {
        rot = tb.rotation[sample.lo];
        t = tb.translation[sample.lo];
        s = tb.scale[sample.lo];
      } else {
        rot =
            slerp(tb.rotation[sample.lo], tb.rotation[sample.hi], sample.alpha);
        t = lerpV3(
            tb.translation[sample.lo], tb.translation[sample.hi], sample.alpha);
        s = lerpV3(tb.scale[sample.lo], tb.scale[sample.hi], sample.alpha);
      }

      result.transforms.push_back({tb.target, composeTransform(rot, t, s)});
    }
  }

  return result;
}

} // namespace tsd::animation
