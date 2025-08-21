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

#include "gpu_decl.h"
#include "gpu_objects.h"

#ifdef USE_MDL
#include <mi/neuraylib/target_code_types.h>
#endif

namespace visrtx {

// Must match the order in which the shaders are pushed to the SBT in
// Renderer.cpp
enum class SurfaceShaderEntryPoints
{
  Initialize = 0,
  EvaluateNextRay,
  EvaluateTint,
  EvaluateOpacity,
  EvaluateEmission,
  Shade,
  Count
};

// Describes the next ray to be traced, as a result of the EvaluateNextRay call
struct NextRay
{
  vec3 direction;
  vec3 contributionWeight;
};

// Matte
struct MatteShadingState
{
  vec3 baseColor;
  float opacity;
};

// Physically Based
struct PhysicallyBasedShadingState
{
  vec3 baseColor;
  vec3 normal;
  float opacity;
  float metallic;
  float roughness;
  float transmission;
  float ior;
  vec3 emission;
};

#ifdef USE_MDL
// See
// https://raytracing-docs.nvidia.com/mdl/api/mi_neuray_example_execution_native.html
//  and
//  https://raytracing-docs.nvidia.com/mdl/api/mi_neuray_example_execution_ptx.html
struct TextureHandler : mi::neuraylib::Texture_handler_base
{
  const visrtx::FrameGPUData *fd;
  // const visrtx::ScreenSample *ss;
  visrtx::DeviceObjectIndex samplers[32];
  unsigned int numSamplers;
};

using ShadingStateMaterial = mi::neuraylib::Shading_state_material;
using ResourceData = mi::neuraylib::Resource_data;

struct MDLShadingState
{
  ShadingStateMaterial state;
  TextureHandler textureHandler;
  ResourceData resData;

  glm::mat3x4 objectToWorld;
  glm::mat3x4 worldToObject;

  // The maximum number of samplers we support.
  // See MDLCompiler.cpp numTextureSpaces and numTextureResults.
  glm::vec4 textureResults[32];
  glm::vec3 textureCoords[4];
  glm::vec3 textureTangentsU[4];
  glm::vec3 textureTangentsV[4];

  bool isFrontFace;

  const char *argBlock;
};
#endif

struct MaterialShadingState
{
  unsigned int callableBaseIndex{~0u};
  ;

  union
  {
    MatteShadingState matte;
    PhysicallyBasedShadingState physicallyBased;
#ifdef USE_MDL
    MDLShadingState mdl;
#endif
  } data;

  VISRTX_DEVICE MaterialShadingState() = default;
};

// #endif

} // namespace visrtx
