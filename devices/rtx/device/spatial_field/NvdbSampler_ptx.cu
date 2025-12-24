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

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/shadingState.h"

using namespace visrtx;

// Fp4 sampler
VISRTX_CALLABLE void __direct_callable__initNvdbSamplerFp4(
    VolumeSamplingState *samplerState,
    const SpatialFieldGPUData *field)
{
  using GridType = nanovdb::Grid<nanovdb::NanoTree<nanovdb::Fp4>>;
  const auto *grid = static_cast<const GridType *>(field->data.nvdbRegular.gridData);
  
  samplerState->nvdbFp4.grid = grid;
  samplerState->nvdbFp4.accessor = grid->getAccessor();
  // Use placement new to construct sampler in-place, as we cannot assign because of 
  // deleted constructor of the nanovdb samplers
  new (&samplerState->nvdbFp4.sampler) NvdbSamplerState<nanovdb::Fp4>::SamplerType(
      nanovdb::math::createSampler<1>(samplerState->nvdbFp4.accessor));
  samplerState->nvdbFp4.scale = 
      (nanovdb::Vec3f(grid->indexBBox().dim()) - nanovdb::Vec3f(1.0f))
      / nanovdb::Vec3f(grid->indexBBox().dim());
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbFp4(
    const VolumeSamplingState *samplerState,
    const vec3 *location)
{
  const auto &state = samplerState->nvdbFp4;
  return state.sampler(state.grid->worldToIndexF(
      nanovdb::Vec3f(location->x, location->y, location->z)) * state.scale);
}

// Fp8 sampler
VISRTX_CALLABLE void __direct_callable__initNvdbSamplerFp8(
    VolumeSamplingState *samplerState,
    const SpatialFieldGPUData *field)
{
  using GridType = nanovdb::Grid<nanovdb::NanoTree<nanovdb::Fp8>>;
  const auto *grid = static_cast<const GridType *>(field->data.nvdbRegular.gridData);
  
  samplerState->nvdbFp8.grid = grid;
  samplerState->nvdbFp8.accessor = grid->getAccessor();
  // Use placement new to construct sampler in-place, as we cannot assign because of 
  // deleted constructor of the nanovdb samplers
  new (&samplerState->nvdbFp8.sampler) NvdbSamplerState<nanovdb::Fp8>::SamplerType(
      nanovdb::math::createSampler<1>(samplerState->nvdbFp8.accessor));
  samplerState->nvdbFp8.scale = 
      (nanovdb::Vec3f(grid->indexBBox().dim()) - nanovdb::Vec3f(1.0f))
      / nanovdb::Vec3f(grid->indexBBox().dim());
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbFp8(
    const VolumeSamplingState *samplerState,
    const vec3 *location)
{
  const auto &state = samplerState->nvdbFp8;
  return state.sampler(state.grid->worldToIndexF(
      nanovdb::Vec3f(location->x, location->y, location->z)) * state.scale);
}

// Fp16 sampler
VISRTX_CALLABLE void __direct_callable__initNvdbSamplerFp16(
    VolumeSamplingState *samplerState,
    const SpatialFieldGPUData *field)
{
  using GridType = nanovdb::Grid<nanovdb::NanoTree<nanovdb::Fp16>>;
  const auto *grid = static_cast<const GridType *>(field->data.nvdbRegular.gridData);
  
  samplerState->nvdbFp16.grid = grid;
  samplerState->nvdbFp16.accessor = grid->getAccessor();
  // Use placement new to construct sampler in-place, as we cannot assign because of 
  // deleted constructor of the nanovdb samplers
  new (&samplerState->nvdbFp16.sampler) NvdbSamplerState<nanovdb::Fp16>::SamplerType(
      nanovdb::math::createSampler<1>(samplerState->nvdbFp16.accessor));
  samplerState->nvdbFp16.scale = 
      (nanovdb::Vec3f(grid->indexBBox().dim()) - nanovdb::Vec3f(1.0f))
      / nanovdb::Vec3f(grid->indexBBox().dim());
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbFp16(
    const VolumeSamplingState *samplerState,
    const vec3 *location)
{
  const auto &state = samplerState->nvdbFp16;
  return state.sampler(state.grid->worldToIndexF(
      nanovdb::Vec3f(location->x, location->y, location->z)) * state.scale);
}

// FpN sampler
VISRTX_CALLABLE void __direct_callable__initNvdbSamplerFpN(
    VolumeSamplingState *samplerState,
    const SpatialFieldGPUData *field)
{
  using GridType = nanovdb::Grid<nanovdb::NanoTree<nanovdb::FpN>>;
  const auto *grid = static_cast<const GridType *>(field->data.nvdbRegular.gridData);
  
  samplerState->nvdbFpN.grid = grid;
  samplerState->nvdbFpN.accessor = grid->getAccessor();
  // Use placement new to construct sampler in-place, as we cannot assign because of 
  // deleted constructor of the nanovdb samplers
  new (&samplerState->nvdbFpN.sampler) NvdbSamplerState<nanovdb::FpN>::SamplerType(
      nanovdb::math::createSampler<1>(samplerState->nvdbFpN.accessor));
  samplerState->nvdbFpN.scale = 
      (nanovdb::Vec3f(grid->indexBBox().dim()) - nanovdb::Vec3f(1.0f))
      / nanovdb::Vec3f(grid->indexBBox().dim());
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbFpN(
    const VolumeSamplingState *samplerState,
    const vec3 *location)
{
  const auto &state = samplerState->nvdbFpN;
  return state.sampler(state.grid->worldToIndexF(
      nanovdb::Vec3f(location->x, location->y, location->z)) * state.scale);
}

// Float sampler
VISRTX_CALLABLE void __direct_callable__initNvdbSamplerFloat(
    VolumeSamplingState *samplerState,
    const SpatialFieldGPUData *field)
{
  using GridType = nanovdb::Grid<nanovdb::NanoTree<float>>;
  const auto *grid = static_cast<const GridType *>(field->data.nvdbRegular.gridData);
  
  samplerState->nvdbFloat.grid = grid;
  samplerState->nvdbFloat.accessor = grid->getAccessor();
  // Use placement new to construct sampler in-place
  new (&samplerState->nvdbFloat.sampler) NvdbSamplerState<float>::SamplerType(
      nanovdb::math::createSampler<1>(samplerState->nvdbFloat.accessor));
  samplerState->nvdbFloat.scale = 
      (nanovdb::Vec3f(grid->indexBBox().dim()) - nanovdb::Vec3f(1.0f))
      / nanovdb::Vec3f(grid->indexBBox().dim());
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbFloat(
    const VolumeSamplingState *samplerState,
    const vec3 *location)
{
  const auto &state = samplerState->nvdbFloat;
  return state.sampler(state.grid->worldToIndexF(
      nanovdb::Vec3f(location->x, location->y, location->z)) * state.scale);
}
