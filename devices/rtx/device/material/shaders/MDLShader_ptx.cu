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

#include "gpu/evalMaterialParameters.h"
#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/shadingState.h"

#include <anari/anari_cpp/ext/linalg.h>
#include <mi/neuraylib/target_code_types.h>
#include <optix_device.h>
#include <glm/ext/matrix_float2x4.hpp>
#include <glm/ext/vector_float4.hpp>

using namespace visrtx;

// No derivatives yet
using BsdfInitFunc = mi::neuraylib::Bsdf_init_function;
using BsdfSampleFunc = mi::neuraylib::Bsdf_sample_function;
using BsdfEvaluateFunc = mi::neuraylib::Bsdf_evaluate_function;
using BsdfPdfFunc = mi::neuraylib::Bsdf_pdf_function;
using BsdfAuxiliaryFunc = mi::neuraylib::Bsdf_auxiliary_function;

using EdfEvaluateFunc = mi::neuraylib::Edf_evaluate_function;

//
using OpacityExprFunc = mi::neuraylib::Material_function<float>::Type;
using TransmissionExprFunc = mi::neuraylib::Material_function<vec3>::Type;
using EmissionIntensityExprFunc = mi::neuraylib::Material_function<vec3>::Type;

using ShadingStateMaterial = mi::neuraylib::Shading_state_material;
using ResourceData = mi::neuraylib::Resource_data;

using BsdfSampleData = mi::neuraylib::Bsdf_sample_data;
using BsdfEvaluateData =
    mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE>;
using BsdfPdfData = mi::neuraylib::Bsdf_pdf_data;
using BsdfAuxiliaryData =
    mi::neuraylib::Bsdf_auxiliary_data<mi::neuraylib::DF_HSM_NONE>;

using BsdfIsThinWalled = bool(
    const ShadingStateMaterial *, const ResourceData *, const char *);

VISRTX_CALLABLE BsdfInitFunc mdlInit;
VISRTX_CALLABLE BsdfSampleFunc mdlBsdf_sample;
VISRTX_CALLABLE BsdfEvaluateFunc mdlBsdf_evaluate;
VISRTX_CALLABLE BsdfPdfFunc mdlBsdf_pdf;
VISRTX_CALLABLE BsdfAuxiliaryFunc mdlBsdf_auxiliary;
VISRTX_CALLABLE BsdfIsThinWalled mdl_isThinWalled;

VISRTX_CALLABLE OpacityExprFunc mdlOpacity;
VISRTX_CALLABLE TransmissionExprFunc mdlTransmission;
VISRTX_CALLABLE EdfEvaluateFunc mdlEmission_evaluate;
VISRTX_CALLABLE EmissionIntensityExprFunc mdlEmissionIntensity;

// Signature must match the call inside shaderMDLSurface in MDLShader.cuh.
VISRTX_CALLABLE void __direct_callable__init(MDLShadingState *shadingState,
    const FrameGPUData *fd,
    const SurfaceHit *hit,
    const MaterialGPUData::MDL *md)
{
  auto position = hit->hitpoint;
  auto Ns = hit->Ns;
  auto Ng = hit->Ng;
  auto tU = hit->tU;
  auto tV = hit->tV;

  // One texture space per ANARI attribute0..3; matches kNumTextureSpaces in
  // libmdl/MDLBackendConfig.h. Tangent frame is the geometry's; multi-UV
  // materials reuse it for every slot (no per-attribute tangent track yet).
  shadingState->textureCoords[0] =
      readAttributeValue(MaterialAttribute::ATTRIB_0, *hit);
  shadingState->textureCoords[1] =
      readAttributeValue(MaterialAttribute::ATTRIB_1, *hit);
  shadingState->textureCoords[2] =
      readAttributeValue(MaterialAttribute::ATTRIB_2, *hit);
  shadingState->textureCoords[3] =
      readAttributeValue(MaterialAttribute::ATTRIB_3, *hit);

  shadingState->textureTangentsU[0] = tU;
  shadingState->textureTangentsU[1] = tU;
  shadingState->textureTangentsU[2] = tU;
  shadingState->textureTangentsU[3] = tU;

  shadingState->textureTangentsV[0] = tV;
  shadingState->textureTangentsV[1] = tV;
  shadingState->textureTangentsV[2] = tV;
  shadingState->textureTangentsV[3] = tV;

  shadingState->state.animation_time = 0.0f;
  shadingState->state.geom_normal = bit_cast<float3>(Ng);
  shadingState->state.normal = bit_cast<float3>(Ns);
  shadingState->state.position = bit_cast<float3>(position);
  shadingState->state.meters_per_scene_unit = 1.0f;
  shadingState->state.object_id = hit->objID;
  shadingState->state.object_to_world =
      reinterpret_cast<const float4 *>(&hit->instance->objectToWorld);
  shadingState->state.world_to_object =
      reinterpret_cast<const float4 *>(&hit->instance->worldToObject);
  shadingState->state.ro_data_segment = nullptr;
  shadingState->state.text_coords =
      reinterpret_cast<const float3 *>(shadingState->textureCoords);
  shadingState->state.text_results =
      reinterpret_cast<float4 *>(shadingState->textureResults);
  shadingState->state.tangent_u =
      reinterpret_cast<const float3 *>(shadingState->textureTangentsU);
  shadingState->state.tangent_v =
      reinterpret_cast<const float3 *>(shadingState->textureTangentsV);

  // Resources shared by all mdl calls. The sampler table is shared with the
  // material descriptor (md->samplers lives in GPU global memory for the
  // lifetime of the material), so the handler holds a pointer rather than an
  // inline copy.
  shadingState->textureHandler.vtable = nullptr;
  shadingState->textureHandler.fd = fd;
  shadingState->textureHandler.samplers = md->samplers;
  shadingState->textureHandler.numSamplers = md->numSamplers;
  shadingState->resData = {nullptr, &shadingState->textureHandler};

  // Front facing for transmission
  shadingState->isFrontFace = hit->isFrontFace;

  // Argument block
  shadingState->argBlock = md->argBlock;

  // Init
  mdlInit(&shadingState->state, &shadingState->resData, shadingState->argBlock);
}

// Signature must match the call inside shaderMDLSurface in MDLShader.cuh.
VISRTX_CALLABLE
vec3 __direct_callable__shadeSurface(const MDLShadingState *shadingState,
    const SurfaceHit *hit,
    const LightSample *lightSample,
    const vec3 *outgoingDir)
{
  // Eval
  const float cos_theta =
      dot(*outgoingDir, normalize(make_vec3(shadingState->state.normal)));
  if (cos_theta > 0.0f) {
    BsdfEvaluateData eval_data = {};
    if (shadingState->isFrontFace) {
      eval_data.ior1 = make_float3(1.0f, 1.0f, 1.0f);
      eval_data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
    } else {
      eval_data.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
      eval_data.ior2 = make_float3(1.0f, 1.0f, 1.0f);
    }
    eval_data.k1 = make_float3(normalize(*outgoingDir));
    eval_data.k2 = make_float3(normalize(lightSample->dir));

    mdlBsdf_evaluate(&eval_data,
        &shadingState->state,
        &shadingState->resData,
        shadingState->argBlock);

    auto radiance_over_pdf = lightSample->radiance / lightSample->pdf;
    auto contrib = radiance_over_pdf
        * (make_vec3(eval_data.bsdf_diffuse)
            + make_vec3(eval_data.bsdf_glossy));

    return contrib;
  }

  return vec3(0.0f, 0.0f, 0.0f);
}

// Signature must match the call inside shaderMDLSurface in MDLShader.cuh.
VISRTX_CALLABLE
NextRay __direct_callable__nextRay(
    const MDLShadingState *shadingState, const Ray *ray, RandState *rs)
{
  // Sample
  BsdfSampleData sample_data = {};
  if (shadingState->isFrontFace) {
    sample_data.ior1 = make_float3(1.0f, 1.0f, 1.0f);
    sample_data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
  } else {
    sample_data.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
    sample_data.ior2 = make_float3(1.0f, 1.0f, 1.0f);
  }
  sample_data.k1 = make_float3(-ray->dir);
  sample_data.xi = make_float4(
      pcg_uniform(rs), pcg_uniform(rs), pcg_uniform(rs), pcg_uniform(rs));

  mdlBsdf_sample(&sample_data,
      &shadingState->state,
      &shadingState->resData,
      shadingState->argBlock);

  const vec3 direction(sample_data.k2.x, sample_data.k2.y, sample_data.k2.z);
  const vec3 N = normalize(make_vec3(shadingState->state.normal));
  const uint32_t flags = dot(ray->dir, N) * dot(direction, N) > 0.0f
      ? NEXT_RAY_CONTINUES_THROUGH_SURFACE
      : NEXT_RAY_NONE;

  // Env-MIS solid-angle pdf for the sampled direction, matching
  // __direct_callable__evaluatePdf so the balance heuristic partitions to 1.
  // A specular (delta) lobe can't be evaluated by NEE, and a through-surface
  // continuation is past the NEE hemisphere gate — both report +inf so the BSDF
  // escape owns the environment (w_bsdf = 1). Glossy/diffuse reflections report
  // the finite sampling pdf and are MIS-combined with NEE.
  const bool isSpecular =
      (sample_data.event_type & mi::neuraylib::BSDF_EVENT_SPECULAR) != 0;
  const float pdf =
      (isSpecular || (flags & NEXT_RAY_CONTINUES_THROUGH_SURFACE))
      ? INFINITY
      : sample_data.pdf;
  return NextRay{direction,
      vec3(sample_data.bsdf_over_pdf.x,
          sample_data.bsdf_over_pdf.y,
          sample_data.bsdf_over_pdf.z),
      pdf,
      flags};
}

// Signature must match the call inside shaderMDLSurface in MDLShader.cuh.
// Base color for the albedo AOV / denoiser guide: the BSDF auxiliary albedo
// (diffuse + glossy) from the compiled material. Evaluated at normal incidence
// (k1 = shading normal) so the guide is view-independent and stable across
// samples.
VISRTX_CALLABLE
vec3 __direct_callable__evaluateTint(const MDLShadingState *shadingState)
{
  BsdfAuxiliaryData aux_data = {};
  aux_data.ior1 = make_float3(1.0f, 1.0f, 1.0f);
  aux_data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
  aux_data.k1 = shadingState->state.normal;

  mdlBsdf_auxiliary(&aux_data,
      &shadingState->state,
      &shadingState->resData,
      shadingState->argBlock);

  return make_vec3(aux_data.albedo_diffuse) + make_vec3(aux_data.albedo_glossy);
}

VISRTX_CALLABLE
float __direct_callable__evaluateOpacity(const MDLShadingState *shadingState)
{
  return mdlOpacity(
      &shadingState->state, &shadingState->resData, shadingState->argBlock);
}

VISRTX_CALLABLE
vec3 __direct_callable__evaluateEmission(
    const MDLShadingState *shadingState, const vec3 *outgoingDir)
{
  mi::neuraylib::Edf_evaluate_data<mi::neuraylib::DF_HSM_NONE> evalData = {};
  evalData.k1 = make_float3(*outgoingDir);

  mdlEmission_evaluate(&evalData,
      &shadingState->state,
      &shadingState->resData,
      shadingState->argBlock);

  vec3 intensity = mdlEmissionIntensity(
      &shadingState->state, &shadingState->resData, shadingState->argBlock);

  // Emitted radiance L(k1) = edf(k1) * intensity. `intensity` is radiant
  // exitance (the material pre-multiplies by PI), and df::diffuse_edf's value is
  // 1/PI, so this yields the authored radiance. The `cos/pdf` factors are the
  // sample-path `edf_over_pdf` quantity, not part of radiance evaluation;
  // including them cancelled to `intensity` and made emission PI x too bright.
  // Matches NVIDIA's df_cuda reference renderer. NOTE: assumes
  // intensity_radiant_exitance mode; intensity_power is not yet handled.
  return make_vec3(evalData.edf) * intensity;
}

VISRTX_CALLABLE
vec3 __direct_callable__evaluateTransmission(
    const MDLShadingState *shadingState)
{
  return mdlTransmission(
      &shadingState->state, &shadingState->resData, shadingState->argBlock);
}

VISRTX_CALLABLE
vec3 __direct_callable__evaluateNormal(const MDLShadingState *shadingState)
{
  return make_vec3(shadingState->state.normal);
}

// Env-MIS BSDF density at `wi` given outgoing `wo` (both world space): the
// balance-heuristic light-side weight. mdlBsdf_evaluate fills `eval_data.pdf`
// with the solid-angle sampling pdf, matching NextRay.pdf in nextRay (MDL's
// evaluate-pdf and sample-pdf are the same density). A pure specular lobe
// evaluates to pdf 0 (NEE can't reach a delta) — consistent with the escape
// owning it via +inf. Mirrors shadeSurface's ior/k1/k2 setup exactly.
VISRTX_CALLABLE float __direct_callable__evaluatePdf(
    const MDLShadingState *shadingState, const vec3 *wo, const vec3 *wi)
{
  BsdfEvaluateData eval_data = {};
  if (shadingState->isFrontFace) {
    eval_data.ior1 = make_float3(1.0f, 1.0f, 1.0f);
    eval_data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
  } else {
    eval_data.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
    eval_data.ior2 = make_float3(1.0f, 1.0f, 1.0f);
  }
  eval_data.k1 = make_float3(normalize(*wo));
  eval_data.k2 = make_float3(normalize(*wi));

  mdlBsdf_evaluate(&eval_data,
      &shadingState->state,
      &shadingState->resData,
      shadingState->argBlock);

  return eval_data.pdf;
}
