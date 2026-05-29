// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ShotCameraRig.h"

#include "tsd/rendering/view/CameraPath.h"

#include <algorithm>
#include <cmath>

namespace tsd::scivis_studio::shot_camera_rig {

const char *toString(CameraInterpolation interpolation)
{
  switch (interpolation) {
  case CameraInterpolation::Hold:
    return "Hold";
  case CameraInterpolation::Linear:
    return "Linear";
  case CameraInterpolation::EaseOut:
    return "Ease Out";
  case CameraInterpolation::EaseIn:
    return "Ease In";
  case CameraInterpolation::EaseOutIn:
    return "Ease Out + In";
  }
  return "Linear";
}

CameraInterpolation interpolationFromString(const std::string &s)
{
  if (s == "Hold")
    return CameraInterpolation::Hold;
  if (s == "Ease Out")
    return CameraInterpolation::EaseOut;
  if (s == "Ease In")
    return CameraInterpolation::EaseIn;
  if (s == "Ease Out + In")
    return CameraInterpolation::EaseOutIn;
  return CameraInterpolation::Linear;
}

ManipulatorState manipulatorStateFromManipulator(
    const tsd::rendering::Manipulator &m)
{
  ManipulatorState state;
  state.orbit.lookat = m.at();
  state.orbit.azeldist =
      tsd::math::float3(m.azel().x, m.azel().y, m.distance());
  state.orbit.fixedDist = m.fixedDistance();
  state.orbit.upAxis = static_cast<int>(m.axis());
  state.orbit.mode = static_cast<int>(m.mode());
  return state;
}

void applyManipulatorState(
    tsd::rendering::Manipulator &m, const ManipulatorState &state)
{
  m.setConfig(state.orbit);
  m.setFixedDistance(state.orbit.fixedDist);
}

void sortKeyframes(ShotCameraRig &rig)
{
  std::stable_sort(rig.keyframes.begin(),
      rig.keyframes.end(),
      [](const CameraKeyframe &a, const CameraKeyframe &b) {
        return a.frame < b.frame;
      });
}

static float lerp(float t, float a, float b)
{
  return a + t * (b - a);
}

static tsd::math::float3 lerpVec3(
    float t, const tsd::math::float3 &a, const tsd::math::float3 &b)
{
  return tsd::math::float3{
      lerp(t, a.x, b.x), lerp(t, a.y, b.y), lerp(t, a.z, b.z)};
}

static float applyInterpolation(CameraInterpolation interpolation, float t)
{
  switch (interpolation) {
  case CameraInterpolation::Hold:
  case CameraInterpolation::Linear:
    return t;
  case CameraInterpolation::EaseOut:
    return t * t;
  case CameraInterpolation::EaseIn:
    return 1.f - (1.f - t) * (1.f - t) * (1.f - t);
  case CameraInterpolation::EaseOutIn:
    return t * t * t * (t * (6.f * t - 15.f) + 10.f);
  }
  return t;
}

ManipulatorState sampleCameraRig(const ShotCameraRig &rig, int frame)
{
  if (rig.keyframes.empty())
    return rig.current;

  if (rig.keyframes.size() == 1)
    return rig.keyframes.front().manipulator;

  auto keyframes = rig.keyframes;
  std::stable_sort(keyframes.begin(),
      keyframes.end(),
      [](const CameraKeyframe &a, const CameraKeyframe &b) {
        return a.frame < b.frame;
      });

  if (frame <= keyframes.front().frame)
    return keyframes.front().manipulator;

  for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
    const auto &a = keyframes[i];
    const auto &b = keyframes[i + 1];
    if (frame > b.frame)
      continue;

    if (a.interpolationToNext == CameraInterpolation::Hold
        || b.frame == a.frame)
      return a.manipulator;

    const float t = static_cast<float>(frame - a.frame)
        / static_cast<float>(b.frame - a.frame);
    const float interpolatedT = applyInterpolation(a.interpolationToNext, t);

    ManipulatorState out;
    out.orbit = a.manipulator.orbit;
    out.orbit.lookat = lerpVec3(
        interpolatedT, a.manipulator.orbit.lookat, b.manipulator.orbit.lookat);
    out.orbit.azeldist = tsd::rendering::lerpAzElDist(interpolatedT,
        a.manipulator.orbit.azeldist,
        b.manipulator.orbit.azeldist);
    out.orbit.fixedDist = lerp(interpolatedT,
        a.manipulator.orbit.fixedDist,
        b.manipulator.orbit.fixedDist);
    out.orbit.upAxis = a.manipulator.orbit.upAxis;
    out.orbit.mode = a.manipulator.orbit.mode;
    return out;
  }

  return keyframes.back().manipulator;
}

} // namespace tsd::scivis_studio::shot_camera_rig
