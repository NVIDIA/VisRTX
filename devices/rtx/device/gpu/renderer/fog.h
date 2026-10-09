// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef __CUDACC__

#include "gpu/gpu_util.h"
#include "gpu/renderer/background.h"

namespace visrtx {

VISRTX_DEVICE vec3 fogColor(
    const FrameGPUData &fd, const uvec2 &pixel, const vec3 &cameraDirection)
{
  const auto &fog = fd.renderer.fog;
  if (fog.mode == FogMode::NONE || fog.colorSource == FogColorSource::CONSTANT)
    return fog.color;
  // Use the very same visible-light sum as the camera miss, before any
  // shading can spawn a reflected ray. Hidden illumination is not a backdrop.
  vec3 hdri;
  if (getBackgroundLight(fd, cameraDirection, hdri))
    return hdri;
  // Screen backdrops are composited at frame resolution, not at jittered or
  // image-region-remapped camera coordinates.
  const vec2 uv = (vec2(pixel) + 0.5f) * fd.fb.invSize;
  return vec3(screenBackground(fd.renderer, uv));
}

// Keep both terms in double: tiny visibility can multiply HDR surface
// radiance, just as a tiny fog fraction can multiply an HDR fog color.
struct FogAppearance
{
  double visibility{1.0};
  double fogFraction{0.0};
};

VISRTX_DEVICE FogAppearance fogAppearance(
    const FrameGPUData &fd, const vec3 &position, const vec3 &cameraOrigin)
{
  const auto &fog = fd.renderer.fog;
  if (fog.mode == FogMode::NONE)
    return {};

  // Widen before subtracting/dotting so finite world coordinates cannot
  // overflow intermediate differences. No world-unit epsilon on intervals.
  const bool rayDistance =
      fog.distanceMetric == FogDistanceMetric::RAY_DISTANCE;
  const glm::dvec3 delta = glm::dvec3(position)
      - glm::dvec3(rayDistance ? cameraOrigin : fd.camera.pos);
  const double distance = rayDistance
      ? glm::length(delta)
      : fmax(0.0, glm::dot(delta, glm::dvec3(fd.camera.dir)));
  if (distance == 0.0)
    return {};

  double fogFraction;
  double visibility;
  if (fog.mode == FogMode::LINEAR) {
    fogFraction =
        fmin(1.0, fmax(0.0, (distance - fog.start) / (fog.end - fog.start)));
    visibility = 1.0 - fogFraction;
  } else {
    const double density = fog.density;
    if (density == 0.0)
      return {};
    const double opticalDistance = density * distance;
    const double exponent = fog.mode == FogMode::EXP2
        ? opticalDistance * opticalDistance
        : opticalDistance;
    visibility = exp(-exponent);
    // Keep tiny fog fractions meaningful with HDR fog colors.
    fogFraction = -expm1(-exponent);
  }
  return {visibility, fogFraction};
}

// Camera appearance only: callers retain coverage and transport weights.
VISRTX_DEVICE vec3 applyFogVisibility(const vec3 &radiance, double visibility)
{
  return vec3(visibility * glm::dvec3(radiance));
}

VISRTX_DEVICE vec3 fogInscatter(
    const FogAppearance &appearance, const vec3 &fogColor)
{
  return vec3(appearance.fogFraction * glm::dvec3(fogColor));
}

VISRTX_DEVICE vec3 fogSurface(const FrameGPUData &fd,
    const vec3 &position,
    const vec3 &cameraOrigin,
    const vec3 &radiance,
    const vec3 &fogColor)
{
  const auto appearance = fogAppearance(fd, position, cameraOrigin);
  return applyFogVisibility(radiance, appearance.visibility)
      + fogInscatter(appearance, fogColor);
}

} // namespace visrtx

#endif // __CUDACC__
