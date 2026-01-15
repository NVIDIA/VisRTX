// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/view/Manipulator.hpp"
// std
#include <algorithm>
#include <vector>

namespace tsd::rendering {

enum class CameraPathInterpolationType
{
  LINEAR,
  SMOOTH_STEP,
  CATMULL_ROM,
  CUBIC_BEZIER
};

struct CameraPathSettings
{
  CameraPathInterpolationType type{CameraPathInterpolationType::LINEAR};
  int framesPerSegment{30}; // frames per pose
  int framesPerSecond{30};
  float tension{0.5f};
};

inline float interpolateCameraPathT(
    float t, CameraPathInterpolationType type, float tension)
{
  switch (type) {
  case CameraPathInterpolationType::LINEAR:
    return t;
  case CameraPathInterpolationType::SMOOTH_STEP:
    return t * t * (3.0f - 2.0f * t);
  case CameraPathInterpolationType::CUBIC_BEZIER: {
    float a = 2.0f - tension;
    if (t < 0.5f) {
      return a * t * t * t;
    } else {
      float f = (2.0f * t - 2.0f);
      return 0.5f * f * f * f + 1.0f;
    }
  }
  case CameraPathInterpolationType::CATMULL_ROM:
  default:
    return t;
  }
}

inline tsd::math::float3 interpolateCameraPathCatmullRom(float t,
    const tsd::math::float3 &p0,
    const tsd::math::float3 &p1,
    const tsd::math::float3 &p2,
    const tsd::math::float3 &p3,
    float tension)
{
  float t2 = t * t;
  float t3 = t2 * t;
  float c = (1.0f - tension) * 0.5f;

  tsd::math::float3 result;
  for (int i = 0; i < 3; ++i) {
    float v0 = (&p0.x)[i];
    float v1 = (&p1.x)[i];
    float v2 = (&p2.x)[i];
    float v3 = (&p3.x)[i];

    (&result.x)[i] = c
        * ((-v0 + 3.0f * v1 - 3.0f * v2 + v3) * t3
            + (2.0f * v0 - 5.0f * v1 + 4.0f * v2 - v3) * t2 + (-v0 + v2) * t
            + 2.0f * v1);
  }

  return result;
}

inline CameraPose sampleCameraPathAt(const std::vector<CameraPose> &poses,
    const CameraPathSettings &settings,
    float t)
{
  auto lerp = [](float t, float a, float b) { return a + t * (b - a); };
  auto lerpVec3 =
      [&lerp](float t, const tsd::math::float3 &a, const tsd::math::float3 &b) {
        return tsd::math::float3{
            lerp(t, a.x, b.x), lerp(t, a.y, b.y), lerp(t, a.z, b.z)};
      };

  if (poses.size() < 2)
    return poses.empty() ? CameraPose{} : poses.front();

  t = std::clamp(t, 0.f, 1.f);
  const float segmentCount = static_cast<float>(poses.size() - 1);
  const float scaled = t * segmentCount;
  const size_t poseIdx =
      std::min(poses.size() - 2, static_cast<size_t>(scaled));
  const float localT = scaled - static_cast<float>(poseIdx);

  const auto &pose0 = poses[poseIdx];
  const auto &pose1 = poses[poseIdx + 1];
  const auto &posePrev = (poseIdx > 0) ? poses[poseIdx - 1] : pose0;
  const auto &poseNext =
      (poseIdx < poses.size() - 2) ? poses[poseIdx + 2] : pose1;

  const float tInterp =
      interpolateCameraPathT(localT, settings.type, settings.tension);

  CameraPose interpPose;
  if (settings.type == CameraPathInterpolationType::CATMULL_ROM) {
    interpPose.lookat = interpolateCameraPathCatmullRom(localT,
        posePrev.lookat,
        pose0.lookat,
        pose1.lookat,
        poseNext.lookat,
        settings.tension);
    interpPose.azeldist = interpolateCameraPathCatmullRom(localT,
        posePrev.azeldist,
        pose0.azeldist,
        pose1.azeldist,
        poseNext.azeldist,
        settings.tension);
    interpPose.fixedDist = lerp(localT, pose0.fixedDist, pose1.fixedDist);
  } else {
    interpPose.lookat = lerpVec3(tInterp, pose0.lookat, pose1.lookat);
    interpPose.azeldist = lerpVec3(tInterp, pose0.azeldist, pose1.azeldist);
    interpPose.fixedDist = lerp(tInterp, pose0.fixedDist, pose1.fixedDist);
  }

  interpPose.upAxis = pose0.upAxis;
  return interpPose;
}

inline void buildCameraPathSamplesByCount(const std::vector<CameraPose> &poses,
    const CameraPathSettings &settings,
    size_t sampleCount,
    std::vector<CameraPose> &outSamples)
{
  outSamples.clear();
  if (poses.size() < 2 || sampleCount == 0)
    return;

  if (sampleCount == 1) {
    outSamples.push_back(poses.front());
    return;
  }

  outSamples.reserve(sampleCount);
  for (size_t i = 0; i < sampleCount; ++i) {
    float t = static_cast<float>(i) / static_cast<float>(sampleCount - 1);
    outSamples.push_back(sampleCameraPathAt(poses, settings, t));
  }
}

inline size_t cameraPathSampleCount(
    const std::vector<CameraPose> &poses, const CameraPathSettings &settings)
{
  if (poses.size() < 2)
    return 0;
  const int framesPerPose = std::max(1, settings.framesPerSegment);
  return poses.size() * static_cast<size_t>(framesPerPose) + 1;
}

inline void buildCameraPathSamples(const std::vector<CameraPose> &poses,
    const CameraPathSettings &settings,
    std::vector<CameraPose> &outSamples)
{
  const size_t count = cameraPathSampleCount(poses, settings);
  buildCameraPathSamplesByCount(poses, settings, count, outSamples);
}

} // namespace tsd::rendering
