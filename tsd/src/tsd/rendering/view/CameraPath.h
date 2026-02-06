// Copyright 2025-2026 NVIDIA Corporation
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
  SMOOTH
};

struct CameraPathSettings
{
  CameraPathInterpolationType type{CameraPathInterpolationType::LINEAR};
  int framesPerSegment{30}; // frames per pose
  int framesPerSecond{30};
  float smoothness{0.5f}; // 0=sharp, 1=very smooth (for SMOOTH mode)
};

// Interpolate azimuth/elevation/distance with proper angle wrapping
inline tsd::math::float3 lerpAzElDist(
    float t, const tsd::math::float3 &a, const tsd::math::float3 &b)
{
  // For angles, compute the shortest rotation path
  auto lerpAngle = [](float t, float a, float b) {
    float diff = b - a;
    // Normalize to [-180, 180]
    while (diff > 180.0f)
      diff -= 360.0f;
    while (diff < -180.0f)
      diff += 360.0f;
    return a + t * diff;
  };

  float azimuth = lerpAngle(t, a.x, b.x);
  float elevation = lerpAngle(t, a.y, b.y);
  float distance = a.z + t * (b.z - a.z);

  return tsd::math::float3{azimuth, elevation, distance};
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

  // Apply global smoothing to the entire path if requested
  // This ensures continuous motion through all poses
  if (settings.type == CameraPathInterpolationType::SMOOTH) {
    // Smooth step function (ease in at start, ease out at end)
    t = t * t * (3.0f - 2.0f * t);
  }

  t = std::clamp(t, 0.f, 1.f);
  const float segmentCount = static_cast<float>(poses.size() - 1);
  const float scaled = t * segmentCount;
  const size_t poseIdx =
      std::min(poses.size() - 2, static_cast<size_t>(scaled));
  const float localT = scaled - static_cast<float>(poseIdx);

  const auto &pose0 = poses[poseIdx];
  const auto &pose1 = poses[poseIdx + 1];

  // Linear interpolation between poses (global smoothing already applied)
  CameraPose interpPose;
  interpPose.lookat = lerpVec3(localT, pose0.lookat, pose1.lookat);
  interpPose.azeldist = lerpAzElDist(localT, pose0.azeldist, pose1.azeldist);
  interpPose.fixedDist = lerp(localT, pose0.fixedDist, pose1.fixedDist);
  interpPose.upAxis = pose0.upAxis;

  return interpPose;
}

// Apply smoothing to the entire path using a simple moving average filter
inline void smoothCameraPath(std::vector<CameraPose> &poses, float smoothness)
{
  if (poses.size() < 3 || smoothness <= 0.0f)
    return;

  // Convert smoothness [0, 1] to window size [1, 5]
  int windowSize = 1 + static_cast<int>(smoothness * 4.0f);
  windowSize = std::clamp(windowSize, 1, 5);

  if (windowSize <= 1)
    return;

  std::vector<CameraPose> smoothed = poses;

  // Apply gaussian-like weighted average
  for (size_t i = 0; i < poses.size(); ++i) {
    tsd::math::float3 avgLookat(0.f);
    tsd::math::float3 avgAzElDist(0.f);
    float avgFixedDist = 0.f;
    float totalWeight = 0.f;

    for (int offset = -windowSize; offset <= windowSize; ++offset) {
      int idx = static_cast<int>(i) + offset;
      if (idx < 0 || idx >= static_cast<int>(poses.size()))
        continue;

      // Gaussian-like weight
      float weight =
          std::exp(-0.5f * (offset * offset) / (windowSize * windowSize));

      avgLookat += poses[idx].lookat * weight;
      avgFixedDist += poses[idx].fixedDist * weight;
      totalWeight += weight;

      // Handle angle wrapping for azimuth/elevation
      if (totalWeight == weight) {
        // First sample - use as-is
        avgAzElDist = poses[idx].azeldist * weight;
      } else {
        // Subsequent samples - use angle-aware interpolation
        tsd::math::float3 normalized = poses[idx].azeldist;

        // Normalize azimuth relative to average
        float azDiff = normalized.x - (avgAzElDist.x / (totalWeight - weight));
        while (azDiff > 180.0f)
          azDiff -= 360.0f;
        while (azDiff < -180.0f)
          azDiff += 360.0f;
        normalized.x = (avgAzElDist.x / (totalWeight - weight)) + azDiff;

        // Normalize elevation relative to average
        float elDiff = normalized.y - (avgAzElDist.y / (totalWeight - weight));
        while (elDiff > 180.0f)
          elDiff -= 360.0f;
        while (elDiff < -180.0f)
          elDiff += 360.0f;
        normalized.y = (avgAzElDist.y / (totalWeight - weight)) + elDiff;

        avgAzElDist += normalized * weight;
      }
    }

    if (totalWeight > 0.f) {
      smoothed[i].lookat = avgLookat / totalWeight;
      smoothed[i].azeldist = avgAzElDist / totalWeight;
      smoothed[i].fixedDist = avgFixedDist / totalWeight;
    }
  }

  poses = smoothed;
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

  // Apply global smoothing if requested
  if (settings.type == CameraPathInterpolationType::SMOOTH) {
    smoothCameraPath(outSamples, settings.smoothness);
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
