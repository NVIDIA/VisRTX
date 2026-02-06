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

#include <cuda_runtime_api.h>
#include "UniformGrid.h"
#include "UniformGridSamplers.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"
#include "gpu/shadingState.h"
#include "gpu/uniformGrid.h"
#ifdef __CUDA_ARCH__
#include "gpu/gpu_util.h"
#endif

namespace visrtx {

__global__ void invalidateRangesGPU(box1 *valueRanges, const ivec3 dims)
{
  size_t threadID = blockIdx.x * size_t(blockDim.x) + threadIdx.x;

  if (threadID >= dims.x * size_t(dims.y) * dims.z)
    return;

  valueRanges[threadID].lower = +1e30f;
  valueRanges[threadID].upper = -1e30f;
}

__global__ void computeMaxOpacitiesGPU(float *maxOpacities,
    const box1 *valueRanges,
    cudaTextureObject_t colorMap,
    size_t numMCs,
    size_t numColors,
    box1 xfRange)
{
  size_t threadID = blockIdx.x * size_t(blockDim.x) + threadIdx.x;

  if (threadID >= numMCs)
    return;

  box1 valueRange = valueRanges[threadID];

  if (valueRange.upper < valueRange.lower) {
    maxOpacities[threadID] = 0.f;
    return;
  }

  valueRange.lower -= xfRange.lower;
  valueRange.lower /= xfRange.upper - xfRange.lower;
  valueRange.upper -= xfRange.lower;
  valueRange.upper /= xfRange.upper - xfRange.lower;

  int lo = glm::clamp(
      int(valueRange.lower * (numColors - 1)), 0, int(numColors - 1));
  int hi = glm::clamp(
      int(valueRange.upper * (numColors - 1)) + 1, 0, int(numColors - 1));

  float maxOpacity = 0.f;
  for (int i = lo; i <= hi; ++i) {
    float tc = (i + .5f) / numColors;
    maxOpacity = fmaxf(maxOpacity, tex1D<::float4>(colorMap, tc).w);
  }
  maxOpacities[threadID] = maxOpacity;
}

template <typename Sampler>
__global__ void buildGridGPU(box1 *valueRanges,
    ivec3 dims,
    box3 worldBounds,
    const SpatialFieldGPUData *sfgd)
{
  Sampler sampler(*sfgd);

  size_t threadID = blockIdx.x * size_t(blockDim.x) + threadIdx.x;

  size_t numVoxels = size_t(dims.x) * dims.y * dims.z;

  if (threadID >= numVoxels)
    return;

  ivec3 voxelID(threadID % dims.x,
      threadID / dims.x % dims.y,
      threadID / (dims.x * dims.y));

  vec3 worldExtend = size(worldBounds);
  vec3 voxelExtend = worldExtend / vec3(dims);
  box3 voxelBounds(worldBounds.lower + vec3(voxelID) * voxelExtend,
      worldBounds.lower + vec3(voxelID) * voxelExtend + voxelExtend);

  // compute the max value of all the cells that can
  // overlap this voxel; splat out the _max_ over the
  // overlapping MCs. (that's essentially a box filter)
  vec3 tcs[8] = {(vec3(voxelID) + vec3(-.5f, -.5f, -.5f)) / vec3(dims),
      (vec3(voxelID) + vec3(+.5f, -.5f, -.5f)) / vec3(dims),
      (vec3(voxelID) + vec3(+.5f, +.5f, -.5f)) / vec3(dims),
      (vec3(voxelID) + vec3(-.5f, +.5f, -.5f)) / vec3(dims),
      (vec3(voxelID) + vec3(-.5f, -.5f, +.5f)) / vec3(dims),
      (vec3(voxelID) + vec3(+.5f, -.5f, +.5f)) / vec3(dims),
      (vec3(voxelID) + vec3(+.5f, +.5f, +.5f)) / vec3(dims),
      (vec3(voxelID) + vec3(-.5f, +.5f, +.5f)) / vec3(dims)};

  float voxelValue = -1e30f;
  for (int i = 0; i < 8; ++i) {
    float retval = sampler(vec3(tcs[i].x, tcs[i].y, tcs[i].z));
    voxelValue = fmaxf(voxelValue, retval);
  }

  // find out which MCs we overlap and splat the value out
  // on the respective ranges
  const ivec3 loMC = projectOnGrid(voxelBounds.lower, dims, worldBounds);
  const ivec3 upMC = projectOnGrid(voxelBounds.upper, dims, worldBounds);

  for (int mcz = loMC.z; mcz <= upMC.z; ++mcz) {
    for (int mcy = loMC.y; mcy <= upMC.y; ++mcy) {
      for (int mcx = loMC.x; mcx <= upMC.x; ++mcx) {
        const ivec3 mcID(mcx, mcy, mcz);
#ifdef __CUDA_ARCH__
        atomicMinf(&valueRanges[linearIndex(mcID, dims)].lower, voxelValue);
        atomicMaxf(&valueRanges[linearIndex(mcID, dims)].upper, voxelValue);
#endif
      }
    }
  }
}

void UniformGrid::init(ivec3 dims, box3 worldBounds)
{
  m_dims = ivec3(iDivUp(dims.x, 16), iDivUp(dims.y, 16), iDivUp(dims.z, 16));
  m_worldBounds = worldBounds;

  size_t numMCs = m_dims.x * size_t(m_dims.y) * m_dims.z;

  cudaFree(m_valueRanges);
  cudaFree(m_maxOpacities);

  cudaMalloc(&m_valueRanges, numMCs * sizeof(box1));
  cudaMalloc(&m_maxOpacities, numMCs * sizeof(float));

  size_t numThreads = 1024;
  invalidateRangesGPU<<<(uint32_t)iDivUp(numMCs, numThreads),
      (uint32_t)numThreads>>>(m_valueRanges, m_dims);
}

void UniformGrid::buildGrid(const SpatialFieldGPUData &sfgd)
{
  size_t numVoxels = size_t(m_dims.x) * m_dims.y * m_dims.z;
  size_t numThreads = 1024;

  // We ned to get the spatialfield gpu data upload, but we don't get
  // to access the framedata store.
  // Let's do a temporary upload so we can do the job.
  SpatialFieldGPUData *sfgdDevice = {};
  cudaMalloc(&sfgdDevice, sizeof(sfgd));
  cudaMemcpy(sfgdDevice, &sfgd, sizeof(sfgd), cudaMemcpyHostToDevice);

  switch (sfgd.samplerCallableIndex) {
  case SbtCallableEntryPoints::SpatialFieldSamplerRegular: {
    buildGridGPU<SpatialFieldSampler<cudaTextureObject_t>>
        <<<iDivUp(numVoxels, numThreads), numThreads>>>(
            m_valueRanges, m_dims, m_worldBounds, sfgdDevice);
    break;
  }
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbFp4: {
    buildGridGPU<NvdbSpatialFieldSampler<nanovdb::Fp4>>
        <<<iDivUp(numVoxels, numThreads), numThreads>>>(
            m_valueRanges, m_dims, m_worldBounds, sfgdDevice);
    break;
  }
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbFp8: {
    buildGridGPU<NvdbSpatialFieldSampler<nanovdb::Fp8>>
        <<<iDivUp(numVoxels, numThreads), numThreads>>>(
            m_valueRanges, m_dims, m_worldBounds, sfgdDevice);
    break;
  }
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbFp16: {
    buildGridGPU<NvdbSpatialFieldSampler<nanovdb::Fp16>>
        <<<iDivUp(numVoxels, numThreads), numThreads>>>(
            m_valueRanges, m_dims, m_worldBounds, sfgdDevice);
    break;
  }
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbFpN: {
    buildGridGPU<NvdbSpatialFieldSampler<nanovdb::FpN>>
        <<<iDivUp(numVoxels, numThreads), numThreads>>>(
            m_valueRanges, m_dims, m_worldBounds, sfgdDevice);
    break;
  }
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbFloat: {
    buildGridGPU<NvdbSpatialFieldSampler<float>>
        <<<iDivUp(numVoxels, numThreads), numThreads>>>(
            m_valueRanges, m_dims, m_worldBounds, sfgdDevice);
    break;
  }
  default:
    break;
  }

  cudaFree(sfgdDevice);
}

void UniformGrid::cleanup()
{
  cudaFree(m_valueRanges);
  cudaFree(m_maxOpacities);

  m_valueRanges = nullptr;
  m_maxOpacities = nullptr;
}

UniformGridData UniformGrid::gpuData() const
{
  UniformGridData grid;
  grid.dims = m_dims;
  grid.worldBounds = m_worldBounds;
  grid.valueRanges = m_valueRanges;
  grid.maxOpacities = m_maxOpacities;
  return grid;
}

void UniformGrid::computeMaxOpacities(
    CUstream stream, cudaTextureObject_t cm, size_t cmSize, box1 cmRange)
{
  size_t numMCs = m_dims.x * size_t(m_dims.y) * m_dims.z;

  size_t numThreads = 1024;
  computeMaxOpacitiesGPU<<<iDivUp(numMCs, numThreads), numThreads, 0, stream>>>(
      m_maxOpacities, m_valueRanges, cm, numMCs, cmSize, cmRange);
}

} // namespace visrtx
