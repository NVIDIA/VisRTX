/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <curand_uniform.h>
#include <device_atomic_functions.h>
#include <algorithm>
#include <cmath>
#include <glm/ext/matrix_float3x3.hpp>
#include <glm/ext/vector_float3.hpp>
#include <limits>
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"

#include <glm/gtx/color_space.hpp>

#include <cub/thread/thread_search.cuh>

namespace visrtx {

// Light sampling result containing direction, distance, radiance and PDF
struct LightSample
{
  vec3 radiance;  // Emitted radiance in direction of hit point (W⋅sr⁻¹⋅m⁻²)
  vec3 dir;       // Unit direction vector from hit point to light sample
  float dist;     // Distance from hit point to light sample
  float pdf;      // Probability density function value for this sample
};

namespace detail {

VISRTX_DEVICE LightSample sampleDirectionalLight(
    const LightGPUData &ld, const mat4 &xfm)
{
  LightSample ls;
  // Transform light direction to world space and negate to get direction TO light
  // (ld.distant.direction points FROM the light source)
  ls.dir = xfmVec(xfm, -ld.distant.direction);
  ls.dist = std::numeric_limits<float>::infinity();
  // For directional lights, irradiance is the amount of light per unit area
  // arriving at the surface (W/m²)
  ls.radiance = ld.color * ld.distant.irradiance;
  // Delta function: directional light has no spatial extent, so PDF = 1
  ls.pdf = 1.f;

  return ls;
}

VISRTX_DEVICE LightSample samplePointLight(
    const LightGPUData &ld, const mat4 &xfm, const Hit &hit)
{
  LightSample ls;
  // Calculate vector from hit point to light position
  ls.dir = xfmPoint(xfm, ld.point.position) - hit.hitpoint;
  ls.dist = length(ls.dir);
  ls.dir /= ls.dist;
  // Apply inverse square law: intensity falls off as 1/r²
  // This converts intensity (W/sr) to radiance at the hit point
  ls.radiance = ld.color * ld.point.intensity / pow2(ls.dist);
  // Delta function: point light has no spatial extent, so PDF = 1
  ls.pdf = 1.f;

  return ls;
}

VISRTX_DEVICE LightSample sampleSphereLight(
    const LightGPUData &ld, const mat4 &xfm, const Hit &hit, RandState &rs)
{
  LightSample ls;
  auto u1 = curand_uniform(&rs);
  auto u2 = curand_uniform(&rs);
  
  // Uniform sampling on unit sphere using Marsaglia's method
  // u1 maps to z-coordinate: z ∈ [-1, 1]
  auto z = 1.f - 2.f * u1;
  // r is the radius in the xy-plane for this z-level
  auto r = sqrtf(std::max(0.f, 1.f - z * z));
  // u2 maps to azimuthal angle: φ ∈ [0, 2π]
  auto phi = 2.f * float(M_PI) * u2;
  auto x = r * cosf(phi);
  auto y = r * sinf(phi);
  
  // Scale by sphere radius to get point on sphere surface
  auto p = vec3(x, y, z) * ld.sphere.radius;
  auto worldSamplePos = xfmPoint(xfm, ld.sphere.position + p);
  ls.dir = worldSamplePos - hit.hitpoint;
  ls.dist = length(ls.dir);
  ls.dir /= ls.dist;
  
  // Sphere emits uniformly in all directions (Lambertian)
  ls.radiance = ld.color * ld.sphere.intensity;
  
  // Convert area PDF to solid angle PDF for proper Monte Carlo integration
  // Area PDF = 1 / (4πr²), but we need solid angle PDF
  // Conversion: pdf_solid_angle = pdf_area * distance² / |cos θ|
  // For sphere: cos θ = dot(surface_normal, -light_direction)
  // Surface normal at sampled point: direction from sphere center to sample point
  auto worldSphereCenter = xfmPoint(xfm, ld.sphere.position);
  auto surfaceNormal = normalize(worldSamplePos - worldSphereCenter);
  auto cosTheta = dot(surfaceNormal, -ls.dir);
  
  if (cosTheta > 0.0f) {
    // Note: For non-uniform scaling transforms, the area calculation would need
    // to account for the transform's effect on surface area (determinant of jacobian)
    // Currently assumes uniform scaling or no scaling of the light geometry
    float areaPdf = 1.f / (4.f * float(M_PI) * ld.sphere.radius * ld.sphere.radius);
    ls.pdf = areaPdf * pow2(ls.dist) / cosTheta;
  } else {
    // Back-facing surface element contributes no light
    ls.radiance = vec3(0.0f);
    ls.pdf = 0.0f;
  }

  return ls;
}

VISRTX_DEVICE LightSample sampleRectLight(
    const LightGPUData &ld, const mat4 &xfm, const Hit &hit, RandState &rs)
{
  LightSample ls;
  auto uv = vec2(curand_uniform(&rs), curand_uniform(&rs));

  // Uniform sampling on rectangle: uv ∈ [0,1]² maps to rectangle
  auto rectangleSample = ld.rect.edge1 * uv.x + ld.rect.edge2 * uv.y;
  auto worldPos = xfmPoint(xfm, ld.rect.position + rectangleSample);
  ls.dir = worldPos - hit.hitpoint;
  ls.dist = length(ls.dir);
  ls.dir /= ls.dist;

  // Calculate rectangle normal and area from cross product
  auto normal = cross(ld.rect.edge1, ld.rect.edge2);
  auto area = length(normal);
  normal = normalize(xfmVec(xfm, normal));

  // Apply Lambert's cosine law: radiance ∝ cos(θ) where θ is angle to normal
  auto cosTheta = dot(normal, -ls.dir);
  
  // Handle front/back face emission based on light configuration
  if (ld.rect.side.back) {
    if (ld.rect.side.front)
      cosTheta = fabsf(cosTheta);  // Both sides: always positive
    else
      cosTheta = -cosTheta;        // Back only: flip to back face
  }
  // Front only: use cosTheta as-is (positive for front face)

  if (cosTheta > 0.0f) {
    // Lambertian emission: radiance scaled by cosine factor
    ls.radiance = ld.color * ld.rect.intensity * cosTheta;
    
    // Convert area PDF to solid angle PDF for proper Monte Carlo integration
    // Area PDF = 1 / area, Solid angle PDF = area_pdf * distance² / |cos θ|
    float areaPdf = 1.0f / area;
    ls.pdf = areaPdf * pow2(ls.dist) / cosTheta;
  } else {
    // No emission toward surfaces facing away from the light
    ls.radiance = vec3(0.0f);
    ls.pdf = 0.0f;
  }

  return ls;
}

VISRTX_DEVICE LightSample sampleSpotLight(
    const LightGPUData &ld, const mat4 &xfm, const Hit &hit)
{
  LightSample ls;
  ls.dir = glm::normalize(xfmPoint(xfm, ld.spot.position) - hit.hitpoint);
  ls.dist = length(ls.dir);
  float spot = dot(normalize(ld.spot.direction), -ls.dir);
  if (spot < ld.spot.cosOuterAngle)
    spot = 0.f;
  else if (spot > ld.spot.cosInnerAngle)
    spot = 1.f;
  else {
    spot = (spot - ld.spot.cosOuterAngle)
        / (ld.spot.cosInnerAngle - ld.spot.cosOuterAngle);
    spot = spot * spot * (3.f - 2.f * spot);
  }
  ls.radiance = ld.color * ld.spot.intensity * spot;
  ls.pdf = 1.f;
  return ls;
}

VISRTX_DEVICE int inverseSampleCDF(const float *cdf, int size, float u)
{
  return cub::LowerBound(cdf, size, u);
}

VISRTX_DEVICE LightSample sampleHDRILight(
    const LightGPUData &ld, const mat4 &xfm, const vec3 &dir)
{
  // Compute the UV coordinates from the direction
  auto thetaPhi = sphericalCoordsFromDirection(ld.hdri.xfm * dir);
  auto uv = glm::vec2(thetaPhi.y, thetaPhi.x)
      / glm::vec2(float(M_PI) * 2.0f, float(M_PI));

  auto radiance = sampleHDRI(ld, uv);
  auto pdf = dot(radiance, {0.2126f, 0.7152f, 0.0722f}) * sinf(thetaPhi.x) * ld.hdri.pdfWeight;

  // Sample the HDRI texture
  LightSample ls;
  ls.dir = xfmVec(xfm, dir);
  ls.dist = std::numeric_limits<float>::infinity();
  ls.radiance = radiance * ld.hdri.scale;
  ls.pdf = pdf;

  return ls;
}

VISRTX_DEVICE LightSample sampleHDRILight(
    const LightGPUData &ld, const mat4 &xfm, const Hit &hit, RandState &rs)
{
  // Row and column sampling
  auto y = inverseSampleCDF(
      ld.hdri.marginalCDF, ld.hdri.size.y, curand_uniform(&rs));
  auto x = inverseSampleCDF(ld.hdri.conditionalCDF + y * ld.hdri.size.x,
      ld.hdri.size.x,
      curand_uniform(&rs));

  auto xy = glm::uvec2(x, y);

#ifdef VISRTX_ENABLE_HDRI_SAMPLING_DEBUG
  if (ld.hdri.samples) {
    atomicInc(ld.hdri.samples + y * ld.hdri.size.x + x, ~0u);
  }
#endif
  auto jitter = glm::vec2(curand_uniform(&rs), curand_uniform(&rs));
  auto uv =
      glm::clamp((glm::vec2(xy) + jitter) / glm::vec2(ld.hdri.size), 0.f, 1.f);

  // And spherical coordinates
  auto thetaPhi = float(M_PI) * glm::vec2(uv.y, 2.0f * (uv.x));

  // Get world direction

  // Compute PDF
  auto radiance = sampleHDRI(ld, uv);
  auto pdf = dot(radiance, {0.2126f, 0.7152f, 0.0722f}) * sinf(thetaPhi.x) * ld.hdri.pdfWeight;

  LightSample ls;
  // ld.hdri.xfm is computed in HDRI.cpp and is made so it is an orthogonal
  // matrix. So transposing is actually the same as inverting. Use RHS
  // multiplication so we don't have to transpose the matrix.
  ls.dir = xfmVec(xfm, sphericalCoordsToDirection(thetaPhi) * ld.hdri.xfm);
  ls.dist = 1e20f;
  ls.radiance = radiance * ld.hdri.scale;
  ls.pdf = pdf;

  return ls;
}

} // namespace detail

VISRTX_DEVICE LightSample sampleLight(
    ScreenSample &ss, const Hit &hit, DeviceObjectIndex idx, const mat4 &xfm)
{
  auto &ld = ss.frameData->registry.lights[idx];

  switch (ld.type) {
  case LightType::DIRECTIONAL:
    return detail::sampleDirectionalLight(ld, xfm);
  case LightType::POINT:
    return detail::samplePointLight(ld, xfm, hit);
  case LightType::SPHERE:
    return detail::sampleSphereLight(ld, xfm, hit, ss.rs);
  case LightType::RECT:
    return detail::sampleRectLight(ld, xfm, hit, ss.rs);
  case LightType::SPOT:
    return detail::sampleSpotLight(ld, xfm, hit);
  case LightType::HDRI:
    return detail::sampleHDRILight(ld, xfm, hit, ss.rs);
  default:
    break;
  }

  return {};
}

} // namespace visrtx
