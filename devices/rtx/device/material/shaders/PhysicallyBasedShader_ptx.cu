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

#include "gpu/adjustNormalForView.h"
#include "gpu/gpu_decl.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"
#include "gpu/shading_api.h"
#include "physicallyBasedSampling.h"

using namespace visrtx;

using namespace visrtx::detail;

//-----------------------------------------------------------------------------
// Helpers
//-----------------------------------------------------------------------------

VISRTX_DEVICE vec3 applyNormalMap(
    const vec3 &tangentSpaceNormal, const SurfaceHit &hit, const vec3 &N)
{
  vec3 T = normalize(hit.tU);
  vec3 B = normalize(hit.tV);
  // Gram-Schmidt to build an orthonormal frame tied to N.
  T = normalize(T - dot(T, N) * N);
  B = normalize(B - dot(B, N) * N - dot(B, T) * T);
  return normalize(T * tangentSpaceNormal.x + B * tangentSpaceNormal.y
      + N * tangentSpaceNormal.z);
}

VISRTX_DEVICE vec3 sampleNormalMap(const FrameGPUData &fd,
    DeviceObjectIndex samplerIdx,
    const SurfaceHit &hit,
    const vec3 &fallback)
{
  if (samplerIdx == ~visrtx::DeviceObjectIndex{0})
    return fallback;
  // Per the ANARI spec the sampler already returns the tangent-space normal;
  // decoding texels (2 * texel - 1) is the app's job, via outTransform and
  // outOffset.
  const vec3 ts = normalize(vec3(evaluateSampler(fd, samplerIdx, hit)));
  const vec3 N = applyNormalMap(ts, hit, hit.Ns);
  // Negated comparison catches NaN (zero-valued samples, zero-summed tangents)
  // as well as zero-length results — fall back to the geometric normal so the
  // shading frame is always usable.
  return (dot(N, N) > 1e-12f) ? N : hit.Ng;
}

// Charlie distribution (Estevez-Kulla 2017) for sheen.
VISRTX_DEVICE float charlieD(float NdotH, float alpha)
{
  const float invAlpha = 1.0f / fmaxf(alpha, 1e-4f);
  const float sin2 = fmaxf(0.0f, 1.0f - NdotH * NdotH);
  return (2.0f + invAlpha) * powf(sin2, 0.5f * invAlpha) / (kTwoPi);
}

// Ashikhmin visibility term (Neubelt-Pettineo variant) used with Charlie D.
VISRTX_DEVICE float charlieV(float NdotV, float NdotL)
{
  return 1.0f / (4.0f * (NdotV + NdotL - NdotV * NdotL) + 1e-6f);
}

//-----------------------------------------------------------------------------
// Initialize shading state from material parameters
//-----------------------------------------------------------------------------

VISRTX_CALLABLE void __direct_callable__init(
    PhysicallyBasedShadingState *shadingState,
    const FrameGPUData *fd,
    const SurfaceHit *hit,
    const MaterialGPUData::PhysicallyBased *md)
{
  const vec4 color = getMaterialParameter(*fd, md->baseColor, *hit);
  const float opacity = getMaterialParameter(*fd, md->opacity, *hit).x;
  shadingState->baseColor = vec3(color);

  const vec3 N = sampleNormalMap(*fd, md->normalSampler, *hit, hit->Ns);
  shadingState->shadingNormal = N;
  shadingState->viewAdjustedNormal = adjustNormalForView(hit->Ng, hit->V, N);

  shadingState->opacity =
      adjustedMaterialOpacity(color.w * opacity, md->alphaMode, md->cutoff);
  shadingState->eta = hit->isFrontFace ? 1.0f / md->ior : md->ior;
  shadingState->metallic = getMaterialParameter(*fd, md->metallic, *hit).x;
  shadingState->roughness = getMaterialParameter(*fd, md->roughness, *hit).x;
  shadingState->emission = vec3(getMaterialParameter(*fd, md->emissive, *hit));
  shadingState->transmission =
      getMaterialParameter(*fd, md->transmission, *hit).x;

  shadingState->occlusion =
      md->occlusionSampler == ~visrtx::DeviceObjectIndex{0}
      ? 1.0f
      : evaluateSampler(*fd, md->occlusionSampler, *hit).x;

  shadingState->specular = getMaterialParameter(*fd, md->specular, *hit).x;
  shadingState->specularColor =
      vec3(getMaterialParameter(*fd, md->specularColor, *hit));

  shadingState->clearcoat = getMaterialParameter(*fd, md->clearcoat, *hit).x;
  shadingState->clearcoatRoughness =
      getMaterialParameter(*fd, md->clearcoatRoughness, *hit).x;
  const vec3 Nc =
      sampleNormalMap(*fd, md->clearcoatNormalSampler, *hit, hit->Ns);
  shadingState->viewAdjustedClearcoatNormal =
      adjustNormalForView(hit->Ng, hit->V, Nc);

  shadingState->thickness = getMaterialParameter(*fd, md->thickness, *hit).x;
  shadingState->attenuationDistance = md->attenuationDistance;
  shadingState->attenuationColor = md->attenuationColor;

  shadingState->sheenColor =
      vec3(getMaterialParameter(*fd, md->sheenColor, *hit));
  shadingState->sheenRoughness =
      getMaterialParameter(*fd, md->sheenRoughness, *hit).x;

  shadingState->iridescence =
      getMaterialParameter(*fd, md->iridescence, *hit).x;
  shadingState->iridescenceIor = md->iridescenceIor;
  shadingState->iridescenceThickness =
      getMaterialParameter(*fd, md->iridescenceThickness, *hit).x;
}

//-----------------------------------------------------------------------------
// Simple accessors
//-----------------------------------------------------------------------------

VISRTX_CALLABLE vec3 __direct_callable__evaluateTint(
    const PhysicallyBasedShadingState *shadingState)
{
  return shadingState->baseColor;
}

VISRTX_CALLABLE float __direct_callable__evaluateOpacity(
    const PhysicallyBasedShadingState *shadingState)
{
  return shadingState->opacity;
}

VISRTX_CALLABLE vec3 __direct_callable__evaluateEmission(
    const PhysicallyBasedShadingState *shadingState, const vec3 *outgoingDir)
{
  return shadingState->emission;
}

VISRTX_CALLABLE vec3 __direct_callable__evaluateTransmission(
    const PhysicallyBasedShadingState *shadingState)
{
  return computeTransmissionFilter(shadingState);
}

VISRTX_CALLABLE vec3 __direct_callable__evaluateNormal(
    const PhysicallyBasedShadingState *shadingState)
{
  return shadingState->shadingNormal;
}

//-----------------------------------------------------------------------------
// NEE shading: base (diffuse + GGX specular) + clearcoat + sheen
//-----------------------------------------------------------------------------

VISRTX_CALLABLE vec3 __direct_callable__evalBsdf(
    const PhysicallyBasedShadingState *state, const vec3 *wo, const vec3 *wi)
{
  const vec3 N = state->viewAdjustedNormal;
  const vec3 V = *wo;
  const vec3 L = *wi;

  const float NdotL = dot(N, L);
  // Negated form so a NaN NdotL takes this early-out — NaN compares false
  // to everything, so `NdotL <= 0.0f` would let it pass through.
  if (!(NdotL > 0.0f))
    return vec3(0.0f);

  const vec3 H = normalize(L + V);
  const float NdotH = fmaxf(dot(N, H), 0.0f);
  const float NdotV = fmaxf(dot(N, V), 1e-6f);
  const float VdotH = fmaxf(dot(V, H), 0.0f);

  // Base F0 / F90. Specular uses Fresnel at the microfacet (VdotH); the
  // diffuse weight uses Fresnel at NdotV (Frostbite/Disney convention) so
  // evalBsdf and nextRay's diffuse split agree regardless of light dir.
  const vec3 F0 = computeF0(state);
  const vec3 F90 = computeF90(state);
  const vec3 F = evalFresnelWithIridescence(state, F0, F90, VdotH);
  const vec3 Fdiff = evalFresnelWithIridescence(state, F0, F90, NdotV);

  // Base GGX specular lobe.
  const float alpha = fmaxf(pow2(state->roughness), 1e-4f);
  const float alpha2 = alpha * alpha;
  const float D = ggxD(NdotH, alpha2);
  const float G2 = smithG2GGX(NdotV, fmaxf(NdotL, 1e-6f), alpha2);
  const vec3 specularBRDF = (F * D * G2) / (4.0f * NdotV * fmaxf(NdotL, 1e-6f));

  // Diffuse lobe (energy-balanced against specular, attenuated by occlusion
  // and transmission; metals have no diffuse).
  const vec3 diffuseColor =
      glm::mix(state->baseColor, vec3(0.0f), state->metallic);
  const vec3 diffuseBRDF = (vec3(1.0f) - Fdiff) * kInvPi * diffuseColor
      * state->occlusion * (1.0f - state->transmission);

  vec3 base = diffuseBRDF + specularBRDF;

  // Clearcoat: a second GGX lobe with its own normal and roughness, Fresnel-
  // attenuating the base layer at both view and light angles.
  if (state->clearcoat > 0.0f) {
    const vec3 Nc = state->viewAdjustedClearcoatNormal;
    const float NcDotV = fmaxf(dot(Nc, V), 1e-6f);
    const float NcDotL = fmaxf(dot(Nc, L), 0.0f);
    const float NcDotH = fmaxf(dot(Nc, H), 0.0f);
    const float FcV =
        CLEARCOAT_F0 + (1.0f - CLEARCOAT_F0) * pow5(1.0f - NcDotV);
    const float FcL =
        CLEARCOAT_F0 + (1.0f - CLEARCOAT_F0) * pow5(1.0f - NcDotL);
    const float alphaC = fmaxf(pow2(state->clearcoatRoughness), 1e-4f);
    const float alphaC2 = alphaC * alphaC;
    const float Dc = ggxD(NcDotH, alphaC2);
    const float Gc = smithG2GGX(NcDotV, fmaxf(NcDotL, 1e-6f), alphaC2);
    // The lobe's Fresnel belongs at the half-vector: H is the facet that
    // actually mirrors V into L, so F(V.H) is what that facet reflects. H
    // depends only on L and V, so the shared base-layer VdotH is the right
    // angle here even though the coat tilts on its own normal Nc, which
    // enters only through Dc(Nc.H) and Gc(Nc.V, Nc.L). The macro-normal
    // FcV/FcL are the *layer* terms — right for attnV/attnL below and for
    // the lobe-pick probability (ccProb in pbrBsdfPdf, clearcoatPick in
    // __direct_callable__nextRay, both on F(Nc.V)), wrong here. This is also
    // what MDL's custom_curve_layer evaluates, and what the sampled weight in
    // __direct_callable__nextRay already used.
    const float FcH = CLEARCOAT_F0 + (1.0f - CLEARCOAT_F0) * pow5(1.0f - VdotH);
    const float clearcoatLobe =
        (FcH * Dc * Gc) / (4.0f * NcDotV * fmaxf(NcDotL, 1e-6f));

    const float attnV = 1.0f - state->clearcoat * FcV;
    const float attnL = 1.0f - state->clearcoat * FcL;
    base = base * attnV * attnL;
    base +=
        vec3(state->clearcoat * clearcoatLobe) * NcDotL / fmaxf(NdotL, 1e-6f);
  }

  // Sheen: Charlie distribution + Ashikhmin visibility, added on top of the
  // base layer without energy compensation (simple but consistent with the
  // glTF reference for basic setups).
  if (glm::any(glm::greaterThan(state->sheenColor, vec3(0.0f)))) {
    const float alphaS = fmaxf(pow2(state->sheenRoughness), 1e-4f);
    const float Ds = charlieD(NdotH, alphaS);
    const float Vs = charlieV(NdotV, fmaxf(NdotL, 1e-6f));
    base += state->sheenColor * Ds * Vs;
  }

  return base * NdotL;
}

//-----------------------------------------------------------------------------
// Next-ray importance sampling: stochastic alpha, Fresnel-aware lobe pick,
// GGX VNDF reflection/refraction, plus a clearcoat lobe sampled with
// probability equal to its view-angle Fresnel weight. Sheen is NEE-only.
//
// Lobe selection is V-only at the diffuse/specular split (so the sampling
// density is the closed form pbrBsdfPdf evaluates); the reflect/transmit split
// within the specular lobe uses the microfacet Fresnel F(VdotH), which TIR
// folds entirely into reflection. NextRay.pdf is filled from pbrBsdfPdf.
//-----------------------------------------------------------------------------

VISRTX_CALLABLE NextRay __direct_callable__nextRay(
    const PhysicallyBasedShadingState *state,
    const Ray *ray,
    RandState *rs,
    const NextRayQmc *qmc)
{
  return samplePhysicallyBased(state, -ray->dir, rs, qmc);
}

//-----------------------------------------------------------------------------
// BSDF sampling pdf at (wo, wi) for environment MIS — the same closed-form
// density nextRay reports in NextRay.pdf, so NEE-side and escape-side weights
// agree. Both directions are world space.
//-----------------------------------------------------------------------------

VISRTX_CALLABLE float __direct_callable__evaluatePdf(
    const PhysicallyBasedShadingState *state, const vec3 *wo, const vec3 *wi)
{
  return pbrBsdfPdf(state, *wo, *wi);
}
