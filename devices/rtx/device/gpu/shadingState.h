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

#include "gpu_decl.h"
#include "gpu_math.h"
#include "gpu_objects.h"

#ifdef USE_MDL
#include <mi/neuraylib/target_code_types.h>
#endif

// nanovdb
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Math.h>
#include <nanovdb/math/SampleFromVoxels.h>

// cuda
#include <texture_types.h>

namespace visrtx {

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
  vec3 normal;
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
  visrtx::DeviceObjectIndex samplers[32];
  unsigned int numSamplers;
};

using ShadingStateMaterial = mi::neuraylib::Shading_state_material;
using ResourceData = mi::neuraylib::Resource_data;

struct alignas(8) MDLShadingState
{
  const char *argBlock;

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
};
#endif

struct MaterialShadingState
{
  unsigned int callableBaseIndex{~0u};

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

// Structured Regular Sampler State
struct StructuredRegularSamplerState
{
  cudaTextureObject_t texObj;
  vec3 origin;
  vec3 invDims;
  vec3 invSpacing;
  vec3 offset;
};

// Structured Rectilinear Sampler State
struct StructuredRectilinearSamplerState
{
  cudaTextureObject_t texObj;
  vec3 dims;
  vec3 offset;
  cudaTextureObject_t axisLUT[3];
  vec3 axisBoundsMin;
  vec3 axisBoundsMax;
  vec3 invAvgVoxelSpacing;
};

// NanoVDB Sampler States
template <typename T>
struct NvdbRegularSamplerState
{
  using GridType = nanovdb::Grid<nanovdb::NanoTree<T>>;
  using AccessorType = typename GridType::AccessorType;
  using NearestSamplerType = nanovdb::math::SampleFromVoxels<AccessorType, 0>;
  using LinearSamplerType = nanovdb::math::SampleFromVoxels<AccessorType, 1>;

  const GridType *grid;
  AccessorType accessor;
  union
  {
    NearestSamplerType nearestSampler;
    LinearSamplerType linearSampler;
  };
  nanovdb::Vec3f offsetDown;
  nanovdb::Vec3f offsetUp;
  nanovdb::Vec3f scale;
  nanovdb::Vec3f indexMin;
  nanovdb::Vec3f indexMax;
  SpatialFieldFilter filter;
};

// NanoVDB Rectilinear Sampler States
template <typename T>
struct NvdbRectilinearSamplerState
{
  using GridType = nanovdb::Grid<nanovdb::NanoTree<T>>;
  using AccessorType = typename GridType::AccessorType;
  using NearestSamplerType = nanovdb::math::SampleFromVoxels<AccessorType, 0>;
  using LinearSamplerType = nanovdb::math::SampleFromVoxels<AccessorType, 1>;

  const GridType *grid;
  AccessorType accessor;
  union
  {
    NearestSamplerType nearestSampler;
    LinearSamplerType linearSampler;
  };
  nanovdb::Vec3f offsetDown;
  nanovdb::Vec3f offsetUp;
  nanovdb::Vec3f scaleDown;
  nanovdb::Vec3f scaleUp;
  nanovdb::Vec3f indexMin;
  nanovdb::Vec3f indexMax;
  cudaTextureObject_t axisLUT[3];
  nanovdb::Vec3f invAvgVoxelSize;
  SpatialFieldFilter filter;
};

// Analytical field sampler states (used by devices/visrtx analytical fields)
#include "spatial_field/AnalyticalFieldData.h"

struct VolumeSamplingState
{
  VISRTX_DEVICE VolumeSamplingState() {};

  union
  {
    StructuredRegularSamplerState structuredRegular;
    NvdbRegularSamplerState<nanovdb::Fp4> nvdbFp4;
    NvdbRegularSamplerState<nanovdb::Fp8> nvdbFp8;
    NvdbRegularSamplerState<nanovdb::Fp16> nvdbFp16;
    NvdbRegularSamplerState<nanovdb::FpN> nvdbFpN;
    NvdbRegularSamplerState<float> nvdbFloat;
    StructuredRectilinearSamplerState structuredRectilinear;
    NvdbRectilinearSamplerState<nanovdb::Fp4> nvdbRectilinearFp4;
    NvdbRectilinearSamplerState<nanovdb::Fp8> nvdbRectilinearFp8;
    NvdbRectilinearSamplerState<nanovdb::Fp16> nvdbRectilinearFp16;
    NvdbRectilinearSamplerState<nanovdb::FpN> nvdbRectilinearFpN;
    NvdbRectilinearSamplerState<float> nvdbRectilinearFloat;
    // Generic analytical field sampler - dispatches by type at runtime
    AnalyticalFieldGPUData analytical;
  };
};

// #endif

} // namespace visrtx
