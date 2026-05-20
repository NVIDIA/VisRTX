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
#include <limits>
#include "UniformGrid.h"
#include "UniformGridAccessors.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"

namespace visrtx {

__global__ void computeOpacityBoundsGPU(float *minOpacities,
    float *maxOpacities,
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
    minOpacities[threadID] = 0.f;
    maxOpacities[threadID] = 0.f;
    return;
  }

  const float xfRangeSize = xfRange.upper - xfRange.lower;
  if (xfRangeSize <= 0.f) {
    minOpacities[threadID] = 0.f;
    maxOpacities[threadID] = 0.f;
    return;
  }

  float normalizedLo = (valueRange.lower - xfRange.lower) / xfRangeSize;
  float normalizedHi = (valueRange.upper - xfRange.lower) / xfRangeSize;

  const float tfEntrySize = 1.0f / float(numColors);
  normalizedLo -= tfEntrySize;
  normalizedHi += tfEntrySize;

  int lo =
      glm::clamp(int(normalizedLo * (numColors - 1)), 0, int(numColors - 1));
  int hi = glm::clamp(
      int(normalizedHi * (numColors - 1)) + 1, 0, int(numColors - 1));

  // The scan range plus 1-bin TF margin and 1-voxel field margin makes both
  // bounds conservative (max ≥ true max, min ≤ true min) for the
  // linear-filter TF lookup over the cell.
  float minOpacity = 1.f;
  float maxOpacity = 0.f;
  for (int i = lo; i <= hi; ++i) {
    float tc = (i + .5f) / numColors;
    float a = tex1D<::float4>(colorMap, tc).w;
    minOpacity = fminf(minOpacity, a);
    maxOpacity = fmaxf(maxOpacity, a);
  }

  minOpacities[threadID] = minOpacity;
  maxOpacities[threadID] = maxOpacity;
}

template <typename VoxelAccessor>
__global__ void computeValueRangesGPU(box1 *valueRanges,
    ivec3 mcDims,
    ivec3 fieldDims,
    const SpatialFieldGPUData *sfgd)
{
  VoxelAccessor accessor(*sfgd);

  size_t threadID = blockIdx.x * size_t(blockDim.x) + threadIdx.x;

  size_t numMCs = size_t(mcDims.x) * mcDims.y * mcDims.z;

  if (threadID >= numMCs)
    return;

  ivec3 mcID(threadID % mcDims.x,
      threadID / mcDims.x % mcDims.y,
      threadID / (mcDims.x * mcDims.y));

  // Field voxel range covered by this macrocell, with 1-voxel margin
  // to account for trilinear interpolation across macrocell boundaries
  const vec3 normLo = vec3(mcID) / vec3(mcDims);
  const vec3 normHi = vec3(mcID + ivec3(1)) / vec3(mcDims);
  const ivec3 voxelLo = glm::max(
      ivec3(0), ivec3(glm::floor(normLo * vec3(fieldDims))) - ivec3(1));
  const ivec3 voxelHi = glm::min(
      fieldDims - ivec3(1), ivec3(glm::ceil(normHi * vec3(fieldDims))));

  float lo = std::numeric_limits<float>::infinity();
  float hi = -std::numeric_limits<float>::infinity();

  for (int iz = voxelLo.z; iz <= voxelHi.z; ++iz) {
    for (int iy = voxelLo.y; iy <= voxelHi.y; ++iy) {
      for (int ix = voxelLo.x; ix <= voxelHi.x; ++ix) {
        float val = accessor(ix, iy, iz);
        if (!isnan(val) && !isinf(val)) {
          lo = fminf(lo, val);
          hi = fmaxf(hi, val);
        }
      }
    }
  }

  if (lo <= hi) {
    valueRanges[threadID].lower = lo;
    valueRanges[threadID].upper = hi;
  } else {
    valueRanges[threadID].lower = std::numeric_limits<float>::infinity();
    valueRanges[threadID].upper = -std::numeric_limits<float>::infinity();
  }
}

size_t UniformGrid::numCells() const
{
  return m_dims.x * size_t(m_dims.y) * m_dims.z;
}

void UniformGrid::init(ivec3 dims, box3 worldBounds)
{
  m_fieldDims = dims;
  m_dims = ivec3(iDivUp(dims.x, MACROCELL_SIZE),
      iDivUp(dims.y, MACROCELL_SIZE),
      iDivUp(dims.z, MACROCELL_SIZE));
  m_worldBounds = worldBounds;

  size_t n = numCells();

  cudaFree(m_valueRanges);
  cudaFree(m_minOpacities);
  cudaFree(m_maxOpacities);

  cudaMalloc(&m_valueRanges, n * sizeof(box1));
  cudaMalloc(&m_minOpacities, n * sizeof(float));
  cudaMalloc(&m_maxOpacities, n * sizeof(float));
}

void UniformGrid::computeValueRanges(const SpatialFieldGPUData &sfgd)
{
  size_t numMCs = numCells();
  size_t numThreads = 1024;

  // Temporary device upload — we don't have access to the framedata store
  SpatialFieldGPUData *sfgdDevice = {};
  cudaMalloc(&sfgdDevice, sizeof(sfgd));
  cudaMemcpy(sfgdDevice, &sfgd, sizeof(sfgd), cudaMemcpyHostToDevice);

#define LAUNCH_BUILD_GRID(Accessor)                                            \
  computeValueRangesGPU<Accessor><<<iDivUp(numMCs, numThreads), numThreads>>>( \
      m_valueRanges, m_dims, m_fieldDims, sfgdDevice)

  switch (sfgd.samplerCallableIndex) {
  case SbtCallableEntryPoints::SpatialFieldSamplerRegular:
    LAUNCH_BUILD_GRID(SpatialFieldAccessor<cudaTextureObject_t>);
    break;
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbFp4:
    LAUNCH_BUILD_GRID(NvdbSpatialFieldAccessor<nanovdb::Fp4>);
    break;
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbFp8:
    LAUNCH_BUILD_GRID(NvdbSpatialFieldAccessor<nanovdb::Fp8>);
    break;
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbFp16:
    LAUNCH_BUILD_GRID(NvdbSpatialFieldAccessor<nanovdb::Fp16>);
    break;
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbFpN:
    LAUNCH_BUILD_GRID(NvdbSpatialFieldAccessor<nanovdb::FpN>);
    break;
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbFloat:
    LAUNCH_BUILD_GRID(NvdbSpatialFieldAccessor<float>);
    break;
  case SbtCallableEntryPoints::SpatialFieldSamplerRectilinear:
    LAUNCH_BUILD_GRID(StructuredRectilinearAccessor);
    break;
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFp4:
    LAUNCH_BUILD_GRID(NvdbRectilinearSpatialFieldAccessor<nanovdb::Fp4>);
    break;
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFp8:
    LAUNCH_BUILD_GRID(NvdbRectilinearSpatialFieldAccessor<nanovdb::Fp8>);
    break;
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFp16:
    LAUNCH_BUILD_GRID(NvdbRectilinearSpatialFieldAccessor<nanovdb::Fp16>);
    break;
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFpN:
    LAUNCH_BUILD_GRID(NvdbRectilinearSpatialFieldAccessor<nanovdb::FpN>);
    break;
  case SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFloat:
    LAUNCH_BUILD_GRID(NvdbRectilinearSpatialFieldAccessor<float>);
    break;
  default:
    break;
  }

#undef LAUNCH_BUILD_GRID

  cudaFree(sfgdDevice);
}

void UniformGrid::cleanup()
{
  cudaFree(m_valueRanges);
  cudaFree(m_minOpacities);
  cudaFree(m_maxOpacities);

  m_valueRanges = nullptr;
  m_minOpacities = nullptr;
  m_maxOpacities = nullptr;
}

UniformGridData UniformGrid::gpuData() const
{
  UniformGridData grid;
  grid.dims = m_dims;
  grid.worldBounds = m_worldBounds;
  grid.valueRanges = m_valueRanges;
  grid.minOpacities = m_minOpacities;
  grid.maxOpacities = m_maxOpacities;
  return grid;
}

void UniformGrid::computeOpacityBounds(
    CUstream stream, cudaTextureObject_t cm, size_t cmSize, box1 cmRange)
{
  size_t n = numCells();
  size_t numThreads = 1024;
  computeOpacityBoundsGPU<<<iDivUp(n, numThreads), numThreads, 0, stream>>>(
      m_minOpacities, m_maxOpacities, m_valueRanges, cm, n, cmSize, cmRange);
}

} // namespace visrtx
