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

#include "evalMaterialParameters.h"
#include "gpu/gpu_objects.h"
#include "shadingState.h"

#include "shaders/MDLShader.cuh"
#include "shaders/MatteShader.cuh"
#include "shaders/PhysicallyBasedShader.cuh"

namespace visrtx {

VISRTX_DEVICE void materialInitShading(MaterialShadingState *shadingState,
    const FrameGPUData &fd,
    const MaterialGPUData &md,
    const SurfaceHit &hit)
{
  shadingState->materialType = md.materialType;
  switch (md.materialType) {
  case MaterialType::MATTE: {
    vec4 color = getMaterialParameter(fd, md.matte.color, hit);
    float opacity = getMaterialParameter(fd, md.matte.opacity, hit).x;

    shadingState->matte = {
        vec3(color),
        adjustedMaterialOpacity(
            color.w * opacity, md.matte.alphaMode, md.matte.cutoff),
    };
    break;
  }
  case MaterialType::PHYSICALLYBASED: {
    vec4 color = getMaterialParameter(fd, md.physicallyBased.baseColor, hit);
    float opacity = getMaterialParameter(fd, md.physicallyBased.opacity, hit).x;

    shadingState->physicallyBased = {
        vec3(color),
        adjustedMaterialOpacity(color.w * opacity,
            md.physicallyBased.alphaMode,
            md.physicallyBased.cutoff),
        getMaterialParameter(fd, md.physicallyBased.metallic, hit).x,
        getMaterialParameter(fd, md.physicallyBased.roughness, hit).x,
        md.physicallyBased.ior,
    };
    break;
  }
  case MaterialType::MDL: {
    mdlInitShading(&shadingState->mdl, fd, hit, md.mdl);
    break;
  }
  default: {
    break;
  }
  }
}

VISRTX_DEVICE vec3 materialEvaluateTint(
    const MaterialShadingState &shadingState)
{
  switch (shadingState.materialType) {
  case MaterialType::MATTE: {
    return shadingState.matte.baseColor;
  }
  case MaterialType::PHYSICALLYBASED: {
    return shadingState.physicallyBased.baseColor;
  }
  case MaterialType::MDL: {
    return mdlEvaluateTint(shadingState.mdl);
  }
  default: {
    return vec3(0.8f, 0.8f, 0.8f); // Default tint color
  }
  }
}

VISRTX_DEVICE float materialEvaluateOpacity(
    const MaterialShadingState &shadingState)
{
  switch (shadingState.materialType) {
  case MaterialType::MATTE: {
    return shadingState.matte.opacity;
  }
  case MaterialType::PHYSICALLYBASED: {
    return shadingState.physicallyBased.opacity;
  }
  case MaterialType::MDL: {
    return mdlEvaluateOpacity(shadingState.mdl);
  }
  default: {
    return 1.0f;
  }
  }
}

VISRTX_DEVICE NextRay materialNextRay(const MaterialShadingState &shadingState,
    const ScreenSample &ss,
    const Ray &ray)
{
  switch (shadingState.materialType) {
  case MaterialType::MDL: {
    return mdlNextRay(shadingState.mdl, ray, ss);
  }
  default: {
    return NextRay{vec4(0.0f), vec4(0.0f)};
  }
  }
}

VISRTX_DEVICE vec3 materialShadeSurface(
    const MaterialShadingState &shadingState,
    const SurfaceHit &hit,
    const LightSample &lightSample,
    const vec3 &outgoingDir)
{
  switch (shadingState.materialType) {
  case MaterialType::MATTE: {
    return matteShadeSurface(shadingState.matte, hit, lightSample, outgoingDir);
  }
  case MaterialType::PHYSICALLYBASED: {
    return physicallyBasedShadeSurface(
        shadingState.physicallyBased, hit, lightSample, outgoingDir);
  }
  case MaterialType::MDL: {
    return mdlShadeSurface(shadingState.mdl, hit, lightSample, outgoingDir);
  }
  default: {
    return vec3(0.0f, 0.0f, 0.0f);
  }
  }
}

} // namespace visrtx
