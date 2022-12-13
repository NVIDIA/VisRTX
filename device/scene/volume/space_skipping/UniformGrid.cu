/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "UniformGrid.h"
#include "gpu/gpu_math.h"

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
    maxOpacity = fmaxf(maxOpacity, tex1D<float4>(colorMap, tc).w);
  }
  maxOpacities[threadID] = maxOpacity;
}

void UniformGrid::init(ivec3 dims, box3 worldBounds)
{
  m_dims = dims;
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
