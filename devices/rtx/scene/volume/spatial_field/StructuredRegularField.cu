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

#include "StructuredRegularField.h"
#include "gpu/uniformGrid.h"
#ifdef __CUDA_ARCH__
#include "gpu/gpu_util.h"
#endif

namespace visrtx {

__global__ void buildGridGPU(
    cudaTextureObject_t data, ivec3 dims, UniformGridData grid)
{
  size_t threadID = blockIdx.x * size_t(blockDim.x) + threadIdx.x;

  size_t numVoxels = (dims.x - 1) * size_t(dims.y - 1) * (dims.z - 1);

  if (threadID >= numVoxels)
    return;

  ivec3 voxelID(threadID % (dims.x - 1),
      threadID / (dims.x - 1) % (dims.y - 1),
      threadID / ((dims.x - 1) * (dims.y - 1)));

  vec3 worldExtend = size(grid.worldBounds);
  vec3 voxelExtend = worldExtend / vec3(dims - 1);
  box3 voxelBounds(grid.worldBounds.lower + vec3(voxelID) * voxelExtend,
      grid.worldBounds.lower + vec3(voxelID) * voxelExtend + voxelExtend);

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
    float retval = tex3D<float>(data, tcs[i].x, tcs[i].y, tcs[i].z);
    voxelValue = fmaxf(voxelValue, retval);
  }

  // find out which MCs we overlap and splat the value out
  // on the respective ranges
  const ivec3 loMC =
      projectOnGrid(voxelBounds.lower, grid.dims, grid.worldBounds);
  const ivec3 upMC =
      projectOnGrid(voxelBounds.upper, grid.dims, grid.worldBounds);

  for (int mcz = loMC.z; mcz <= upMC.z; ++mcz) {
    for (int mcy = loMC.y; mcy <= upMC.y; ++mcy) {
      for (int mcx = loMC.x; mcx <= upMC.x; ++mcx) {
        const ivec3 mcID(mcx, mcy, mcz);
#ifdef __CUDA_ARCH__
        atomicMinf(
            &grid.valueRanges[linearIndex(mcID, grid.dims)].lower, voxelValue);
        atomicMaxf(
            &grid.valueRanges[linearIndex(mcID, grid.dims)].upper, voxelValue);
#endif
      }
    }
  }
}

void StructuredRegularField::buildGrid()
{
  auto dims = m_data->size();
  ivec3 gridDims(iDivUp(dims.x, 16), iDivUp(dims.y, 16), iDivUp(dims.z, 16));
  m_uniformGrid.init(gridDims, bounds());

  size_t numVoxels = (dims.x - 1) * size_t(dims.y - 1) * (dims.z - 1);

  size_t numThreads = 1024;
  buildGridGPU<<<iDivUp(numVoxels, numThreads),
      numThreads,
      0,
      deviceState()->stream>>>(
      m_textureObject, ivec3(dims.x, dims.y, dims.z), m_uniformGrid.gpuData());
}

} // namespace visrtx
