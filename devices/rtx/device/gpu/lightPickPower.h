/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Pick Power: a scalar estimate of a light's emitted power (relative flux),
// used only to weight the power-proportional Light Pick. Approximation here
// costs variance, never bias, so the estimates are luminance-weighted flux with
// deliberately coarse handling of transforms and cone falloff. Every light with
// a nonzero contribution must return a strictly positive value.

#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"

namespace visrtx {

namespace detail {

VISRTX_HOST_DEVICE float pickLuminance(const vec3 &c)
{
  return glm::dot(c, vec3(0.2126f, 0.7152f, 0.0722f));
}

// Isotropic area-scale factor of an affine transform: an object-space area A
// maps to roughly A * |det|^(2/3) in world space. Exact under uniform scale;
// an approximation under shear/non-uniform scale. For Geometry Lights this only
// shifts Pick Power (variance): the sampled pdf is corrected exactly at sample
// time via worldAreaScale. Analytic sphere/rect/ring samplers use object-space
// area with no transform Jacobian, so there the approximation is uncorrected
// (documented in sampleSphereLight).
VISRTX_HOST_DEVICE float affineAreaScale(const mat4 &xfm)
{
  const float det = glm::abs(glm::determinant(mat3(xfm)));
  return powf(det, 2.0f / 3.0f);
}

} // namespace detail

VISRTX_HOST_DEVICE float lightPickPower(
    const LightGPUData &ld, const mat4 &xfm, float sceneRadius)
{
  const float sceneCrossSection = kPi * sceneRadius * sceneRadius;

  switch (ld.type) {
  case LightType::DIRECTIONAL:
    // Irradiance (W/m²) intercepted over the scene cross-section.
    return detail::pickLuminance(ld.color) * ld.distant.irradiance
        * sceneCrossSection;
  case LightType::POINT:
    // Isotropic point light: total flux = 4π · intensity.
    return detail::pickLuminance(ld.color) * ld.point.intensity * 2.0f
        * kTwoPi;
  case LightType::SPHERE: {
    // Lambertian sphere: flux = L · area · π, area = 4πr².
    const float area = kTwoPi * 2.0f * ld.sphere.radius * ld.sphere.radius
        * detail::affineAreaScale(xfm);
    return detail::pickLuminance(ld.color) * ld.sphere.intensity * area * kPi;
  }
  case LightType::RECT: {
    // Lambertian rectangle: flux = L · area · π, doubled if it emits both sides.
    const float area = detail::affineAreaScale(xfm)
        / glm::max(ld.rect.oneOverArea, 1e-8f);
    const float sides = float(ld.rect.side.front + ld.rect.side.back);
    return detail::pickLuminance(ld.color) * ld.rect.intensity * area * kPi
        * sides;
  }
  case LightType::SPOT: {
    // Point light restricted to a cone: flux ≈ intensity · solid angle.
    const float coneSolidAngle = kTwoPi * (1.0f - ld.spot.cosOuterAngle);
    return detail::pickLuminance(ld.color) * ld.spot.intensity * coneSolidAngle;
  }
  case LightType::RING: {
    // Lambertian disk annulus; the cone falloff is ignored for the estimate.
    const float area = detail::affineAreaScale(xfm)
        / glm::max(ld.ring.oneOverArea, 1e-8f);
    return detail::pickLuminance(ld.color) * ld.ring.intensity * area * kPi;
  }
  case LightType::HDRI:
    // The environment's average luminance is approximated as unit; scale and
    // tint carry the only per-light signal until a measured average lands.
    return detail::pickLuminance(ld.color) * ld.hdri.scale * sceneCrossSection;
  case LightType::GEOMETRY: {
    // Double-sided Lambertian surface: flux = L · area · π, doubled for sides.
    const float area = ld.geometry.area * detail::affineAreaScale(xfm);
    return detail::pickLuminance(ld.geometry.radiance) * area * kPi * 2.0f;
  }
  default:
    return 0.0f;
  }
}

} // namespace visrtx
