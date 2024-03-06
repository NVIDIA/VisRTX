/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// This file borrows heavily from 'brdf.glsl' found in the glTF-Sample-Viewer.
//
// https://github.com/KhronosGroup/glTF-Sample-Viewer
//

#pragma once

#include "gpu_math.h"

namespace visrtx {

// Fresnel ////////////////////////////////////////////////////////////////////

// http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
// https://github.com/wdas/brdf/tree/master/src/brdfs
// https://google.github.io/filament/Filament.md.html
//

// The following equation models the Fresnel reflectance term of the spec
// equation (aka F()) Implementation of fresnel from [4], Equation 15
RT_FUNCTION vec3 F_Schlick(vec3 f0, vec3 f90, float VdotH)
{
  return f0 + (f90 - f0) * glm::pow(glm::clamp(1.f - VdotH, 0.f, 1.f), 5.f);
}

RT_FUNCTION float F_Schlick(float f0, float f90, float VdotH)
{
  float x = glm::clamp(1.f - VdotH, 0.f, 1.f);
  float x2 = x * x;
  float x5 = x * x2 * x2;
  return f0 + (f90 - f0) * x5;
}

RT_FUNCTION float F_Schlick(float f0, float VdotH)
{
  float f90 = 1.f; // glm::clamp(50.0 * f0, 0.f, 1.f);
  return F_Schlick(f0, f90, VdotH);
}

RT_FUNCTION vec3 F_Schlick(vec3 f0, float f90, float VdotH)
{
  float x = glm::clamp(1.f - VdotH, 0.f, 1.f);
  float x2 = x * x;
  float x5 = x * x2 * x2;
  return f0 + (f90 - f0) * x5;
}

RT_FUNCTION vec3 F_Schlick(vec3 f0, float VdotH)
{
  float f90 = 1.f; // glm::clamp(glm::dot(f0, vec3(50.f * 0.33f)), 0.f, 1.f);
  return F_Schlick(f0, f90, VdotH);
}

RT_FUNCTION vec3 Schlick_to_F0(vec3 f, vec3 f90, float VdotH)
{
  float x = glm::clamp(1.f - VdotH, 0.f, 1.f);
  float x2 = x * x;
  float x5 = glm::clamp(x * x2 * x2, 0.f, 0.9999f);

  return (f - f90 * x5) / (1.f - x5);
}

RT_FUNCTION float Schlick_to_F0(float f, float f90, float VdotH)
{
  float x = glm::clamp(1.f - VdotH, 0.f, 1.f);
  float x2 = x * x;
  float x5 = glm::clamp(x * x2 * x2, 0.f, 0.9999f);

  return (f - f90 * x5) / (1.f - x5);
}

RT_FUNCTION vec3 Schlick_to_F0(vec3 f, float VdotH)
{
  return Schlick_to_F0(f, vec3(1.f), VdotH);
}

RT_FUNCTION float Schlick_to_F0(float f, float VdotH)
{
  return Schlick_to_F0(f, 1.f, VdotH);
}

// Smith Joint GGX ////////////////////////////////////////////////////////////

// Note: Vis = G / (4 * NdotL * NdotV)
// see Eric Heitz. 2014. Understanding the Masking-Shadowing Function in
// Microfacet-Based BRDFs. Journal of Computer Graphics Techniques, 3 see
// Real-Time Rendering. Page 331 to 336. see
// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
RT_FUNCTION float V_GGX(float NdotL, float NdotV, float alphaRoughness)
{
  float alphaRoughnessSq = alphaRoughness * alphaRoughness;

  float GGXV = NdotL
      * glm::sqrt(NdotV * NdotV * (1.f - alphaRoughnessSq) + alphaRoughnessSq);
  float GGXL = NdotV
      * glm::sqrt(NdotL * NdotL * (1.f - alphaRoughnessSq) + alphaRoughnessSq);

  float GGX = GGXV + GGXL;
  if (GGX > 0.f) {
    return 0.5f / GGX;
  }
  return 0.f;
}

// The following equation(s) model the distribution of microfacet normals across
// the area being drawn (aka D()) Implementation from "Average Irregularity
// Representation of a Roughened Surface for Ray Reflection" by T. S.
// Trowbridge, and K. P. Reitz Follows the distribution function recommended in
// the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
RT_FUNCTION float D_GGX(float NdotH, float alphaRoughness)
{
  float alphaRoughnessSq = alphaRoughness * alphaRoughness;
  float f = (NdotH * NdotH) * (alphaRoughnessSq - 1.0) + 1.0;
  return alphaRoughnessSq / (M_PI * f * f);
}

RT_FUNCTION float lambdaSheenNumericHelper(float x, float alphaG)
{
  float oneMinusAlphaSq = (1.f - alphaG) * (1.f - alphaG);
  float a = glm::mix(21.5473f, 25.3245f, oneMinusAlphaSq);
  float b = glm::mix(3.82987f, 3.32435f, oneMinusAlphaSq);
  float c = glm::mix(0.19823f, 0.16801f, oneMinusAlphaSq);
  float d = glm::mix(-1.97760f, -1.27393f, oneMinusAlphaSq);
  float e = glm::mix(-4.32054f, -4.85967f, oneMinusAlphaSq);
  return a / (1.f + b * glm::pow(x, c)) + d * x + e;
}

RT_FUNCTION float lambdaSheen(float cosTheta, float alphaG)
{
  if (glm::abs(cosTheta) < 0.5f) {
    return glm::exp(lambdaSheenNumericHelper(cosTheta, alphaG));
  } else {
    return glm::exp(2.f * lambdaSheenNumericHelper(0.5f, alphaG)
        - lambdaSheenNumericHelper(1.f - cosTheta, alphaG));
  }
}

RT_FUNCTION float V_Sheen(float NdotL, float NdotV, float sheenRoughness)
{
  sheenRoughness = max(sheenRoughness, 0.000001f); // glm::clamp (0,1]
  float alphaG = sheenRoughness * sheenRoughness;

  return glm::clamp(1.f
          / ((1.f + lambdaSheen(NdotV, alphaG) + lambdaSheen(NdotL, alphaG))
              * (4.f * NdotV * NdotL)),
      0.f,
      1.f);
}

// Sheen //////////////////////////////////////////////////////////////////////

//  See
//  https://github.com/sebavan/glTF/tree/KHR_materials_sheen/extensions/2.0/Khronos/KHR_materials_sheen

// Estevez and Kulla http://www.aconty.com/pdf/s2017_pbs_imageworks_sheen.pdf
RT_FUNCTION float D_Charlie(float sheenRoughness, float NdotH)
{
  sheenRoughness = glm::max(sheenRoughness, 0.000001f); // glm::clamp (0,1]
  float alphaG = sheenRoughness * sheenRoughness;
  float invR = 1.f / alphaG;
  float cos2h = NdotH * NdotH;
  float sin2h = 1.f - cos2h;
  return (2.f + invR) * glm::pow(sin2h, invR * 0.5f) / (2.f * float(M_PI));
}

// https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments
// AppendixB
RT_FUNCTION vec3 BRDF_lambertian(
    vec3 f0, vec3 f90, vec3 diffuseColor, float specularWeight, float VdotH)
{
  // see
  // https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
  return (1.f - specularWeight * F_Schlick(f0, f90, VdotH))
      * (diffuseColor / float(M_PI));
}

// https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments
// AppendixB
RT_FUNCTION vec3 BRDF_lambertianIridescence(vec3 f0,
    vec3 f90,
    vec3 iridescenceFresnel,
    float iridescenceFactor,
    vec3 diffuseColor,
    float specularWeight,
    float VdotH)
{
  // Use the maximum component of the iridescence Fresnel color
  // Maximum is used instead of the RGB value to not get inverse colors for the
  // diffuse BRDF
  vec3 iridescenceFresnelMax = vec3(max(
      max(iridescenceFresnel.r, iridescenceFresnel.g), iridescenceFresnel.b));

  vec3 schlickFresnel = F_Schlick(f0, f90, VdotH);

  // Blend default specular Fresnel with iridescence Fresnel
  vec3 F = mix(schlickFresnel, iridescenceFresnelMax, iridescenceFactor);

  // see
  // https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
  return (1.f - specularWeight * F) * (diffuseColor / float(M_PI));
}

//  https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments
//  AppendixB
RT_FUNCTION vec3 BRDF_specularGGX(vec3 f0,
    vec3 f90,
    float alphaRoughness,
    float specularWeight,
    float VdotH,
    float NdotL,
    float NdotV,
    float NdotH)
{
  vec3 F = F_Schlick(f0, f90, VdotH);
  float Vis = V_GGX(NdotL, NdotV, alphaRoughness);
  float D = D_GGX(NdotH, alphaRoughness);

  return specularWeight * F * Vis * D;
}

RT_FUNCTION vec3 BRDF_specularGGXIridescence(vec3 f0,
    vec3 f90,
    vec3 iridescenceFresnel,
    float alphaRoughness,
    float iridescenceFactor,
    float specularWeight,
    float VdotH,
    float NdotL,
    float NdotV,
    float NdotH)
{
  vec3 F =
      mix(F_Schlick(f0, f90, VdotH), iridescenceFresnel, iridescenceFactor);
  float Vis = V_GGX(NdotL, NdotV, alphaRoughness);
  float D = D_GGX(NdotH, alphaRoughness);

  return specularWeight * F * Vis * D;
}

// GGX Distribution Anisotropic (Same as Babylon.js) //////////////////////////

// https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
// Addenda
RT_FUNCTION float D_GGX_anisotropic(
    float NdotH, float TdotH, float BdotH, float anisotropy, float at, float ab)
{
  float a2 = at * ab;
  vec3 f = vec3(ab * TdotH, at * BdotH, a2 * NdotH);
  float w2 = a2 / glm::dot(f, f);
  return a2 * w2 * w2 / M_PI;
}

// GGX Mask/Shadowing Anisotropic (Same as Babylon.js -
// smithVisibility_GGXCorrelated_Anisotropic) Heitz
// http://jcgt.org/published/0003/02/03/paper.pdf
RT_FUNCTION float V_GGX_anisotropic(float NdotL,
    float NdotV,
    float BdotV,
    float TdotV,
    float TdotL,
    float BdotL,
    float at,
    float ab)
{
  float GGXV = NdotL * length(vec3(at * TdotV, ab * BdotV, NdotV));
  float GGXL = NdotV * length(vec3(at * TdotL, ab * BdotL, NdotL));
  float v = 0.5f / (GGXV + GGXL);
  return glm::clamp(v, 0.f, 1.f);
}

RT_FUNCTION vec3 BRDF_specularGGXAnisotropy(vec3 f0,
    vec3 f90,
    float alphaRoughness,
    float anisotropy,
    vec3 n,
    vec3 v,
    vec3 l,
    vec3 h,
    vec3 t,
    vec3 b)
{
  // Roughness along the anisotropy bitangent is the material roughness, while
  // the tangent roughness increases with anisotropy.
  float at = glm::mix(alphaRoughness, 1.f, anisotropy * anisotropy);
  float ab = glm::clamp(alphaRoughness, 0.001f, 1.f);

  float NdotL = glm::clamp(glm::dot(n, l), 0.f, 1.f);
  float NdotH = glm::clamp(glm::dot(n, h), 0.001f, 1.f);
  float NdotV = glm::dot(n, v);
  float VdotH = glm::clamp(glm::dot(v, h), 0.f, 1.f);

  float V = V_GGX_anisotropic(NdotL,
      NdotV,
      glm::dot(b, v),
      glm::dot(t, v),
      glm::dot(t, l),
      glm::dot(b, l),
      at,
      ab);
  float D = D_GGX_anisotropic(
      NdotH, glm::dot(t, h), glm::dot(b, h), anisotropy, at, ab);

  vec3 F = F_Schlick(f0, f90, VdotH);
  return F * V * D;
}

RT_FUNCTION vec3 BRDF_specularSheen(vec3 sheenColor,
    float sheenRoughness,
    float NdotL,
    float NdotV,
    float NdotH)
{
  float sheenDistribution = D_Charlie(sheenRoughness, NdotH);
  float sheenVisibility = V_Sheen(NdotL, NdotV, sheenRoughness);
  return sheenColor * sheenDistribution * sheenVisibility;
}

} // namespace visrtx
