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
#include <curand.h>
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

using EdfEvaluateFunc = mi::neuraylib::Edf_evaluate_function;

//
using TintExprFunc = mi::neuraylib::Material_function<vec3>::Type;
using OpacityExprFunc = mi::neuraylib::Material_function<float>::Type;
using TransmissionExprFunc = mi::neuraylib::Material_function<vec3>::Type;
using EmissionIntensityExprFunc = mi::neuraylib::Material_function<vec3>::Type;

using ShadingStateMaterial = mi::neuraylib::Shading_state_material;
using ResourceData = mi::neuraylib::Resource_data;

using BsdfSampleData = mi::neuraylib::Bsdf_sample_data;
using BsdfEvaluateData =
    mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE>;
using BsdfPdfData = mi::neuraylib::Bsdf_pdf_data;

using BsdfIsThinWalled = bool(
    const ShadingStateMaterial *, const ResourceData *, const char *);

VISRTX_CALLABLE BsdfInitFunc mdlInit;
VISRTX_CALLABLE BsdfSampleFunc mdlBsdf_sample;
VISRTX_CALLABLE BsdfEvaluateFunc mdlBsdf_evaluate;
VISRTX_CALLABLE BsdfPdfFunc mdlBsdf_pdf;
VISRTX_CALLABLE BsdfIsThinWalled mdl_isThinWalled;

VISRTX_CALLABLE TintExprFunc mdlTint;
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

  shadingState->objectToWorld = hit->objectToWorld;
  shadingState->worldToObject = hit->worldToObject;

  // The number of texture spaces we support. Matching the number of attributes
  // ANARI exposes (4)
  shadingState->textureCoords[0] =
      readAttributeValue(MaterialAttribute::ATTRIB_0, *hit);
  shadingState->textureCoords[1] =
      readAttributeValue(MaterialAttribute::ATTRIB_1, *hit);
  shadingState->textureCoords[2] =
      readAttributeValue(MaterialAttribute::ATTRIB_2, *hit);
  shadingState->textureCoords[3] =
      readAttributeValue(MaterialAttribute::ATTRIB_3, *hit);

  // Take some shortcut for now and use the same tangent space for all texture
  // spaces.
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
      reinterpret_cast<const float4 *>(&shadingState->objectToWorld);
  shadingState->state.world_to_object =
      reinterpret_cast<const float4 *>(&shadingState->worldToObject);
  shadingState->state.ro_data_segment = nullptr;
  shadingState->state.text_coords =
      reinterpret_cast<const float3 *>(shadingState->textureCoords);
  shadingState->state.text_results =
      reinterpret_cast<float4 *>(shadingState->textureResults);
  shadingState->state.tangent_u =
      reinterpret_cast<const float3 *>(shadingState->textureTangentsU);
  shadingState->state.tangent_v =
      reinterpret_cast<const float3 *>(shadingState->textureTangentsV);

  // Resources shared by all mdl calls.
  shadingState->textureHandler.vtable = nullptr;
  shadingState->textureHandler.fd = fd;
  memcpy(shadingState->textureHandler.samplers,
      md->samplers,
      sizeof(md->samplers));
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
  // Before anything, check for opacity. If below, than we just pass through
  if (curand_uniform(rs) > mdlOpacity(&shadingState->state,
          &shadingState->resData,
          shadingState->argBlock)) {
    return NextRay{ray->dir, vec3(1.0f)};
  }

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
  sample_data.xi = make_float4(curand_uniform(rs),
      curand_uniform(rs),
      curand_uniform(rs),
      curand_uniform(rs));

  mdlBsdf_sample(&sample_data,
      &shadingState->state,
      &shadingState->resData,
      shadingState->argBlock);

  return NextRay{vec3(sample_data.k2.x, sample_data.k2.y, sample_data.k2.z),
      vec3(sample_data.bsdf_over_pdf.x,
          sample_data.bsdf_over_pdf.y,
          sample_data.bsdf_over_pdf.z)};
}

// Signature must match the call inside shaderMDLSurface in MDLShader.cuh.
VISRTX_CALLABLE
vec3 __direct_callable__evaluateTint(const MDLShadingState *shadingState)
{
  return mdlTint(
      &shadingState->state, &shadingState->resData, shadingState->argBlock);
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

  return (evalData.pdf > 1e-12f)
      ? make_vec3(evalData.edf) * evalData.cos * intensity / evalData.pdf
      : vec3(0.0f);
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