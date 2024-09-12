/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu/shading_api.h"

using namespace visrtx;

// Signature must match the call inside shaderPhysicallyBasedSurface in
// PhysicallyBasedShader.cuh.
VISRTX_CALLABLE vec4 __direct_callable__evalSurfaceMaterial(
    const FrameGPUData *fd,
    const MaterialGPUData::PhysicallyBased *md,
    const SurfaceHit *hit,
    const vec3 *viewDir,
    const vec3 *lightDir,
    const vec3 *lightIntensity)
{
  const auto matValues = getMaterialValues(*fd, *md, *hit);

  const vec3 H = normalize(*lightDir + *viewDir);
  const float NdotH = dot(hit->Ns, H);
  const float NdotL = dot(hit->Ns, *lightDir);
  const float NdotV = dot(hit->Ns, *viewDir);
  const float VdotH = dot(*viewDir, H);
  const float LdotH = dot(*lightDir, H);

  // Fresnel
  const vec3 f0 =
      glm::mix(vec3(pow2((1.f - matValues.ior) / (1.f + matValues.ior))),
          matValues.baseColor,
          matValues.metallic);
  const vec3 F = f0 + (vec3(1.f) - f0) * pow5(1.f - fabsf(VdotH));

  // Metallic materials don't reflect diffusely:
  const vec3 diffuseColor =
      glm::mix(matValues.baseColor, vec3(0.f), matValues.metallic);

  const vec3 diffuseBRDF =
      (vec3(1.f) - F) * float(M_1_PI) * diffuseColor * fmaxf(0.f, NdotL);

  // Alpha
  const float alpha = pow2(matValues.roughness) * matValues.opacity;

  // GGX microfacet distribution
  const float D = (alpha * alpha * heaviside(NdotH))
      / (float(M_PI) * pow2(NdotH * NdotH * (alpha * alpha - 1.f) + 1.f));

  // Masking-shadowing term
  const float G =
      ((2.f * fabsf(NdotL) * heaviside(LdotH))
          / (fabsf(NdotL)
              + sqrtf(alpha * alpha + (1.f - alpha * alpha) * NdotL * NdotL)))
      * ((2.f * fabsf(NdotV) * heaviside(VdotH))
          / (fabsf(NdotV)
              + sqrtf(alpha * alpha + (1.f - alpha * alpha) * NdotV * NdotV)));

  const float denom = 4.f * fabsf(NdotV) * fabsf(NdotL);
  const vec3 specularBRDF = denom != 0.f ? (F * D * G) / denom : vec3(0.f);

  return {(diffuseBRDF + specularBRDF) * *lightIntensity, matValues.opacity};
}
