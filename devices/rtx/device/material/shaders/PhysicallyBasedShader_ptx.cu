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

#include "gpu/gpu_decl.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"
#include "gpu/shading_api.h"

using namespace visrtx;

// Clearcoat is fixed to IOR 1.5 per glTF KHR_materials_clearcoat (F0 = 0.04).
constexpr float CLEARCOAT_F0 = 0.04f;

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
  const vec3 ts = normalize(evaluateSampler(fd, samplerIdx, hit) * 2.0f - 1.0f);
  const vec3 N = applyNormalMap(ts, hit, hit.Ns);
  // Negated comparison catches NaN (zero-decoded texels, zero-summed tangents)
  // as well as zero-length results — fall back to the geometric normal so the
  // shading frame is always usable.
  return (dot(N, N) > 1e-12f) ? N : hit.Ng;
}

VISRTX_DEVICE float luminance(const vec3 &c)
{
  return dot(c, vec3(0.2126f, 0.7152f, 0.0722f));
}

VISRTX_DEVICE vec3 computeVolumeTransmission(
    const PhysicallyBasedShadingState *state)
{
  if (!(state->thickness > 0.0f && state->attenuationDistance > 0.0f
          && isfinite(state->attenuationDistance)))
    return vec3(1.0f);

  const float k = state->thickness / state->attenuationDistance;
  return vec3(powf(fmaxf(state->attenuationColor.x, 1e-6f), k),
      powf(fmaxf(state->attenuationColor.y, 1e-6f), k),
      powf(fmaxf(state->attenuationColor.z, 1e-6f), k));
}

VISRTX_DEVICE vec3 computeTransmissionFilter(
    const PhysicallyBasedShadingState *state)
{
  const float transmission =
      fmaxf(0.0f, (1.0f - state->metallic) * state->transmission);
  return state->baseColor * transmission * computeVolumeTransmission(state);
}

// Smith Lambda for GGX (common subterm of G1 / G2).
VISRTX_DEVICE float smithLambdaGGX(float NdotX, float alpha2)
{
  const float NdotX2 = NdotX * NdotX;
  const float safe = fmaxf(NdotX2, 1e-8f);
  return 0.5f
      * (-1.0f + sqrtf(fmaxf(0.0f, 1.0f + alpha2 * (1.0f - safe) / safe)));
}

VISRTX_DEVICE float smithG2GGX(float NdotV, float NdotL, float alpha2)
{
  return 1.0f
      / (1.0f + smithLambdaGGX(NdotV, alpha2) + smithLambdaGGX(NdotL, alpha2));
}

VISRTX_DEVICE float smithG1GGX(float NdotV, float alpha2)
{
  return 1.0f / (1.0f + smithLambdaGGX(NdotV, alpha2));
}

VISRTX_DEVICE float ggxD(float NdotH, float alpha2)
{
  // The textbook denom `x·(α²−1) + 1` cancels catastrophically in fp32 once
  // α² is below eps(1) ≈ 1.19e-7 (our α² floor is 1e-8): `α²−1` rounds to
  // exactly −1, and at x=1 the whole denom collapses to 0. The algebraically
  // equivalent `α²·x + (1−x)` has no near-1 subtraction so it stays exact.
  // The fminf clamp keeps `1−x ≥ 0` against dot-product rounding above 1.
  const float NdotH2 = fminf(NdotH * NdotH, 1.0f);
  const float denom = alpha2 * NdotH2 + (1.0f - NdotH2);
  return alpha2 / (kPi * denom * denom);
}

// Heitz 2018 (https://jcgt.org/published/0007/04/01/) visible-normal sampling
// for GGX. Ve is the view direction in local tangent space (+z = normal).
VISRTX_DEVICE vec3 sampleGGXVNDF(
    const vec3 &Ve, float alpha, float u1, float u2)
{
  const vec3 Vh = normalize(vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));
  const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  const vec3 T1 = lensq > 0.0f ? vec3(-Vh.y, Vh.x, 0.0f) * (1.0f / sqrtf(lensq))
                               : vec3(1.0f, 0.0f, 0.0f);
  const vec3 T2 = glm::cross(Vh, T1);
  const float r = sqrtf(u1);
  const float phi = kTwoPi * u2;
  const float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);
  const float s = 0.5f * (1.0f + Vh.z);
  t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;
  const vec3 Nh =
      t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
  return normalize(vec3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
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

// glTF KHR_materials_iridescence thin-film Fresnel (port of the reference
// implementation at github.com/KhronosGroup/glTF-Sample-Renderer). Returns a
// per-channel Fresnel reflectance for a thin film of thickness T sitting on a
// base with Schlick F0. See the spec's Appendix B for the math.
VISRTX_DEVICE vec3 fresnel0ToIor(vec3 F0)
{
  const vec3 s = glm::sqrt(glm::clamp(F0, vec3(0.0f), vec3(0.9999f)));
  return (vec3(1.0f) + s) / (vec3(1.0f) - s);
}

VISRTX_DEVICE vec3 iorToFresnel0(vec3 transmittedIor, float incidentIor)
{
  const vec3 t = (transmittedIor - vec3(incidentIor))
      / (transmittedIor + vec3(incidentIor));
  return t * t;
}

VISRTX_DEVICE float iorToFresnel0(float transmittedIor, float incidentIor)
{
  const float t =
      (transmittedIor - incidentIor) / (transmittedIor + incidentIor);
  return t * t;
}

VISRTX_DEVICE vec3 evalSensitivity(float opd, vec3 shift)
{
  // Approximate spectral sensitivity of the standard observer as three
  // Gaussians (Belcour & Barla 2017, simplified) so the result stays in RGB.
  const float phase = kTwoPi * opd * 1e-9f;
  const vec3 val = vec3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
  const vec3 pos = vec3(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
  const vec3 var = vec3(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);

  vec3 xyz = val * glm::sqrt(kTwoPi * var) * glm::cos(pos * phase + shift)
      * glm::exp(-var * phase * phase);
  xyz.x += 9.7470e-14f * sqrtf(kTwoPi * 4.5282e+09f)
      * cosf(2.2399e+06f * phase + shift.x)
      * expf(-4.5282e+09f * phase * phase);
  xyz /= 1.0685e-7f;

  // sRGB conversion (D65).
  return vec3(3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
      -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
      0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z);
}

VISRTX_DEVICE vec3 evalIridescence(float outsideIor,
    float iridescenceIor,
    float cosTheta1,
    float thickness,
    vec3 baseF0)
{
  // Handle the case where thin-film IOR is close to the outside IOR: return
  // the base Fresnel to avoid division by zero and phase artifacts.
  const float iridescenceIorSafe = fmaxf(iridescenceIor, outsideIor + 1e-4f);

  // Force iridescenceIor > outsideIor (otherwise Snell's law cannot refract).
  const float sinTheta2Sq =
      pow2(outsideIor / iridescenceIorSafe) * (1.0f - cosTheta1 * cosTheta1);
  const float cosTheta2Sq = 1.0f - sinTheta2Sq;
  if (cosTheta2Sq < 0.0f)
    return vec3(1.0f); // Total internal reflection.
  const float cosTheta2 = sqrtf(cosTheta2Sq);

  // First interface: Fresnel between outside and thin film.
  const float R0_12 = iorToFresnel0(iridescenceIorSafe, outsideIor);
  const float R12 = R0_12 + (1.0f - R0_12) * pow5(1.0f - cosTheta1);
  const float T121 = 1.0f - R12;
  const float phi12 = iridescenceIorSafe < outsideIor ? kPi : 0.0f;
  const float phi21 = kPi - phi12;

  // Second interface: film to base.
  const vec3 baseIor =
      fresnel0ToIor(glm::clamp(baseF0, vec3(0.f), vec3(0.9999f)));
  const vec3 R1 = iorToFresnel0(baseIor, iridescenceIorSafe);
  const vec3 R23 = R1 + (vec3(1.0f) - R1) * pow5(1.0f - cosTheta2);
  const vec3 phi23 = vec3(baseIor.x < iridescenceIorSafe ? kPi : 0.0f,
      baseIor.y < iridescenceIorSafe ? kPi : 0.0f,
      baseIor.z < iridescenceIorSafe ? kPi : 0.0f);

  const float opd = 2.0f * iridescenceIorSafe * thickness * cosTheta2;
  const vec3 phi = vec3(phi21) + phi23;

  const vec3 R123 = glm::clamp(R12 * R23, vec3(1e-5f), vec3(0.9999f));
  const vec3 r123 = glm::sqrt(R123);
  const vec3 Rs = pow2(T121) * R23 / (vec3(1.0f) - R123);

  // DC term.
  vec3 C0 = R12 + Rs;
  vec3 I = C0;

  // Higher-order terms.
  vec3 Cm = Rs - T121;
  for (int m = 1; m <= 2; ++m) {
    Cm *= r123;
    const vec3 Sm = 2.0f * evalSensitivity(float(m) * opd, float(m) * phi);
    I += Cm * Sm;
  }

  return glm::max(I, vec3(0.0f));
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
  shadingState->normal = N;

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
  shadingState->useSpecular = md->useSpecular;

  shadingState->clearcoat = getMaterialParameter(*fd, md->clearcoat, *hit).x;
  shadingState->clearcoatRoughness =
      getMaterialParameter(*fd, md->clearcoatRoughness, *hit).x;
  shadingState->clearcoatNormal =
      sampleNormalMap(*fd, md->clearcoatNormalSampler, *hit, hit->Ns);

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
  return shadingState->normal;
}

//-----------------------------------------------------------------------------
// NEE shading: base (diffuse + GGX specular) + clearcoat + sheen
//-----------------------------------------------------------------------------

VISRTX_DEVICE vec3 computeDielectricF0(const PhysicallyBasedShadingState *state)
{
  const float iorF0 = pow2((1.0f - state->eta) / (1.0f + state->eta));
  if (state->useSpecular == 0)
    return vec3(iorF0);
  return glm::min(vec3(iorF0) * state->specularColor, vec3(1.0f))
      * state->specular;
}

VISRTX_DEVICE vec3 computeF0(const PhysicallyBasedShadingState *state)
{
  return glm::mix(
      computeDielectricF0(state), state->baseColor, state->metallic);
}

VISRTX_DEVICE vec3 computeF90(const PhysicallyBasedShadingState *state)
{
  const float dielectricF90 = state->useSpecular == 0 ? 1.0f : state->specular;
  return glm::mix(vec3(dielectricF90), vec3(1.0f), state->metallic);
}

VISRTX_DEVICE vec3 schlickFresnel(vec3 F0, vec3 F90, float VdotH)
{
  return F0 + (F90 - F0) * pow5(1.0f - fabsf(VdotH));
}

VISRTX_DEVICE vec3 evalFresnelWithIridescence(
    const PhysicallyBasedShadingState *state,
    const vec3 &F0,
    const vec3 &F90,
    float cosTheta)
{
  vec3 F = schlickFresnel(F0, F90, cosTheta);
  if (state->iridescence > 0.0f && state->iridescenceThickness > 0.0f) {
    const vec3 iridescent = evalIridescence(
        1.0f, state->iridescenceIor, cosTheta, state->iridescenceThickness, F0);
    F = glm::mix(F, iridescent, state->iridescence);
  }
  return F;
}

VISRTX_CALLABLE vec3 __direct_callable__shadeSurface(
    const PhysicallyBasedShadingState *state,
    const SurfaceHit *hit,
    const LightSample *lightSample,
    const vec3 *outgoingDir)
{
  const vec3 N = state->normal;
  const vec3 V = *outgoingDir;
  const vec3 L = lightSample->dir;

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
  // shadeSurface and nextRay's diffuse split agree regardless of light dir.
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
    const vec3 Nc = state->clearcoatNormal;
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
    const float clearcoatLobe =
        (FcV * Dc * Gc) / (4.0f * NcDotV * fmaxf(NcDotL, 1e-6f));

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

  return base * NdotL * lightSample->radiance / lightSample->pdf;
}

//-----------------------------------------------------------------------------
// Next-ray importance sampling: stochastic alpha, Fresnel-aware lobe pick,
// GGX VNDF reflection/refraction, plus a clearcoat lobe sampled with
// probability equal to its view-angle Fresnel weight. Sheen is NEE-only.
//-----------------------------------------------------------------------------

VISRTX_CALLABLE NextRay __direct_callable__nextRay(
    const PhysicallyBasedShadingState *state, const Ray *ray, RandState *rs)
{
  // Opacity pass-through (stochastic alpha): the ray continues unaltered.
  if (curand_uniform(rs) > state->opacity)
    return NextRay{ray->dir, vec3(1.0f), NEXT_RAY_CONTINUES_THROUGH_SURFACE};

  const vec3 V = -ray->dir;

  // Clearcoat lobe: pick it with probability `clearcoat·FcV(NcDotV)`. This
  // exact weight makes the entry-side attenuation `1 - clearcoat·FcV` cancel
  // the `1/(1-pick)` lobe-pick divisor in the base path below, so the base
  // returns only need the exit-side `1 - clearcoat·FcL` multiplier.
  const vec3 Nc = state->clearcoatNormal;
  const float NcDotV_world = fmaxf(dot(Nc, V), 0.0f);
  const float FcV_world =
      CLEARCOAT_F0 + (1.0f - CLEARCOAT_F0) * pow5(1.0f - NcDotV_world);
  const float clearcoatPick =
      glm::clamp(state->clearcoat * FcV_world, 0.0f, 1.0f);

  if (clearcoatPick > 0.0f && curand_uniform(rs) < clearcoatPick) {
    const mat3 toWorldC = computeOrthonormalBasis(Nc);
    const vec3 VlocalC = glm::transpose(toWorldC) * V;
    if (VlocalC.z <= 0.0f)
      return NextRay{Nc, vec3(0.0f)};
    const float alphaC = fmaxf(pow2(state->clearcoatRoughness), 1e-4f);
    const float alphaC2 = alphaC * alphaC;
    const vec3 HlocalC = sampleGGXVNDF(
        VlocalC, alphaC, curand_uniform(rs), curand_uniform(rs));
    const vec3 LlocalC = glm::reflect(-VlocalC, HlocalC);
    if (LlocalC.z <= 0.0f)
      return NextRay{Nc, vec3(0.0f)};
    const float VdotHc = fmaxf(dot(VlocalC, HlocalC), 0.0f);
    const float Fc =
        CLEARCOAT_F0 + (1.0f - CLEARCOAT_F0) * pow5(1.0f - VdotHc);
    const float G1c = smithG1GGX(VlocalC.z, alphaC2);
    const float G2c = smithG2GGX(VlocalC.z, LlocalC.z, alphaC2);
    // VNDF gives BRDF·cos/pdf = clearcoat·Fc·G2/G1; the clearcoat factor
    // cancels against the matching factor in clearcoatPick.
    const vec3 weight = vec3(state->clearcoat * Fc * G2c / fmaxf(G1c, 1e-8f))
        / fmaxf(clearcoatPick, 1e-8f);
    return NextRay{normalize(toWorldC * LlocalC), weight};
  }

  // Exit-side clearcoat attenuation, applied to every base-path return.
  // `fabsf` handles the transmission case where L points through the surface.
  auto clearcoatExitAttn = [&](const vec3 &Lworld) -> float {
    if (state->clearcoat <= 0.0f)
      return 1.0f;
    const float NcDotL = fabsf(dot(Nc, Lworld));
    const float FcL =
        CLEARCOAT_F0 + (1.0f - CLEARCOAT_F0) * pow5(1.0f - NcDotL);
    return glm::clamp(1.0f - state->clearcoat * FcL, 0.0f, 1.0f);
  };

  const vec3 N = state->normal;
  const mat3 toWorld = computeOrthonormalBasis(N);
  const mat3 toLocal = glm::transpose(toWorld);
  const vec3 Vlocal = toLocal * V;
  if (Vlocal.z <= 0.0f)
    return NextRay{N, vec3(0.0f)};

  const float alpha = fmaxf(pow2(state->roughness), 1e-4f);
  const float alpha2 = alpha * alpha;
  const vec3 Hlocal =
      sampleGGXVNDF(Vlocal, alpha, curand_uniform(rs), curand_uniform(rs));

  const float NdotV = Vlocal.z;
  const float VdotH = fmaxf(dot(Vlocal, Hlocal), 0.0f);

  // Fresnel at the sampled microfacet (specular/transmission split) and at
  // NdotV (diffuse weight) — matches the convention in shadeSurface.
  const vec3 F0 = computeF0(state);
  const vec3 F90 = computeF90(state);
  const vec3 F = evalFresnelWithIridescence(state, F0, F90, VdotH);
  const vec3 Fdiff = evalFresnelWithIridescence(state, F0, F90, NdotV);

  const vec3 Lrefl = glm::reflect(-Vlocal, Hlocal);
  const vec3 Ltrans = glm::refract(-Vlocal, Hlocal, state->eta);
  const vec3 transmissionFilter = computeTransmissionFilter(state);
  const bool hasTransmission = luminance(transmissionFilter) > 0.0f;
  const bool totalInternalReflection =
      hasTransmission && (glm::length(Ltrans) < 1e-6f || Ltrans.z >= 0.0f);

  vec3 reflectEnergy = totalInternalReflection ? vec3(1.0f) : F;
  vec3 transmitEnergy = totalInternalReflection
      ? vec3(0.0f)
      : glm::max(vec3(1.0f) - F, vec3(0.0f)) * transmissionFilter;

  // Diffuse importance: the Lambertian throughput collapses to
  //   (1-F) * baseColor * (1-metallic) * (1-transmission) * occlusion
  // when sampled cosine-weighted (cos / pdf cancels with 1/pi). Mirror the
  // factors used by shadeSurface's diffuseBRDF so the lobe split tracks the
  // BRDF being estimated. TIR has no diffuse share (all energy is reflected).
  const vec3 diffuseEnergy = totalInternalReflection
      ? vec3(0.0f)
      : glm::max(vec3(1.0f) - Fdiff, vec3(0.0f)) * state->baseColor
          * (1.0f - state->metallic) * (1.0f - state->transmission)
          * state->occlusion;

  const float reflectStrength =
      fmaxf(luminance(glm::max(reflectEnergy, vec3(0.0f))), 0.0f);
  const float transmitStrength =
      fmaxf(luminance(glm::max(transmitEnergy, vec3(0.0f))), 0.0f);
  const float diffuseStrength =
      fmaxf(luminance(glm::max(diffuseEnergy, vec3(0.0f))), 0.0f);
  const float combinedStrength =
      reflectStrength + transmitStrength + diffuseStrength;
  if (combinedStrength <= 0.0f)
    return NextRay{N, vec3(0.0f)};

  const float reflectProb = reflectStrength / combinedStrength;
  const float transmitProb = transmitStrength / combinedStrength;
  const float diffuseProb = diffuseStrength / combinedStrength;

  const float u = curand_uniform(rs);
  if (u < reflectProb) {
    if (Lrefl.z <= 0.0f)
      return NextRay{N, vec3(0.0f)};
    const float NdotL = Lrefl.z;
    const float G1 = smithG1GGX(NdotV, alpha2);
    const float G2 = smithG2GGX(NdotV, NdotL, alpha2);
    const vec3 Lworld = normalize(toWorld * Lrefl);
    const vec3 weight = reflectEnergy * (G2 / fmaxf(G1, 1e-8f))
        * clearcoatExitAttn(Lworld) / fmaxf(reflectProb, 1e-8f);
    return NextRay{Lworld, weight};
  }

  if (u < reflectProb + transmitProb) {
    const float NdotL = -Ltrans.z; // L points through the surface.
    const float G1 = smithG1GGX(NdotV, alpha2);
    const float G2 = smithG2GGX(NdotV, NdotL, alpha2);
    const vec3 Lworld = normalize(toWorld * Ltrans);
    const vec3 weight = transmitEnergy * (G2 / fmaxf(G1, 1e-8f))
        * clearcoatExitAttn(Lworld) / fmaxf(transmitProb, 1e-8f);
    return NextRay{Lworld, weight, NEXT_RAY_CONTINUES_THROUGH_SURFACE};
  }

  // Diffuse: sample around the shading normal so pdf=cos/pi matches the BRDF's
  // NdotL (same axis as shadeSurface's diffuse term). Cos and pdf cancel,
  // leaving only the energy term and the lobe-pick divisor.
  const vec3 wi = sampleHemisphere(*rs, N);
  const vec3 weight =
      diffuseEnergy * clearcoatExitAttn(wi) / fmaxf(diffuseProb, 1e-8f);
  return NextRay{wi, weight};
}
