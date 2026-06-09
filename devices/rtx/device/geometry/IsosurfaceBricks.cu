/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "IsosurfaceBricks.h"
#include "gpu/gpu_math.h"
// thrust
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace visrtx {

// Macrocells per brick edge. Larger -> fewer, looser bricks (cheaper/smaller
// BVH, weaker culling); 1 -> one primitive per macrocell. A field with fewer
// than BRICK_SIZE macrocells per axis collapses to a single brick (= the whole
// field bounds), matching the single-primitive path.
static constexpr int BRICK_SIZE = 4;

__global__ void buildBricksGPU(box3 *bricks,
    uint8_t *flags,
    UniformGridData grid,
    const float *isovalues,
    uint32_t numIsovalues,
    ivec3 brickDims)
{
  const size_t b = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  const size_t n = size_t(brickDims.x) * brickDims.y * brickDims.z;
  if (b >= n)
    return;

  const ivec3 brick(b % brickDims.x,
      (b / brickDims.x) % brickDims.y,
      b / (size_t(brickDims.x) * brickDims.y));

  const vec3 gridSize = grid.objectBounds.upper - grid.objectBounds.lower;
  const ivec3 mcBegin = brick * BRICK_SIZE;
  const ivec3 mcEnd = min(mcBegin + ivec3(BRICK_SIZE), grid.dims);

  bool active = false;
  vec3 lo(3.0e38f);
  vec3 hi(-3.0e38f);
  for (int z = mcBegin.z; z < mcEnd.z; ++z) {
    for (int y = mcBegin.y; y < mcEnd.y; ++y) {
      for (int x = mcBegin.x; x < mcEnd.x; ++x) {
        const size_t i =
            x + size_t(grid.dims.x) * (y + size_t(grid.dims.y) * z);
        const box1 vr = grid.valueRanges[i];
        if (!(vr.lower <= vr.upper))
          continue;
        bool cellActive = false;
        for (uint32_t k = 0; k < numIsovalues; ++k) {
          const float iv = isovalues[k];
          if (iv >= vr.lower && iv <= vr.upper) {
            cellActive = true;
            break;
          }
        }
        if (!cellActive)
          continue;
        active = true;
        const ivec3 mc(x, y, z);
        lo = min(lo,
            grid.objectBounds.lower + (vec3(mc) / vec3(grid.dims)) * gridSize);
        hi = max(hi,
            grid.objectBounds.lower
                + (vec3(mc + ivec3(1)) / vec3(grid.dims)) * gridSize);
      }
    }
  }

  flags[b] = active ? uint8_t(1) : uint8_t(0);
  bricks[b] = active ? box3(lo, hi) : box3(vec3(0.f), vec3(0.f));
}

size_t buildIsosurfaceBricks(cudaStream_t stream,
    const UniformGridData &grid,
    const float *isovaluesDev,
    uint32_t numIsovalues,
    DeviceBuffer &outAabbs)
{
  if (grid.valueRanges == nullptr || isovaluesDev == nullptr
      || numIsovalues == 0 || grid.dims.x <= 0 || grid.dims.y <= 0
      || grid.dims.z <= 0)
    return 0;

  const ivec3 brickDims(iDivUp(grid.dims.x, BRICK_SIZE),
      iDivUp(grid.dims.y, BRICK_SIZE),
      iDivUp(grid.dims.z, BRICK_SIZE));
  const size_t n = size_t(brickDims.x) * brickDims.y * brickDims.z;

  thrust::device_vector<box3> dense(n);
  thrust::device_vector<uint8_t> flags(n);

  const size_t threads = 256;
  buildBricksGPU<<<iDivUp(n, threads), threads, 0, stream>>>(
      thrust::raw_pointer_cast(dense.data()),
      thrust::raw_pointer_cast(flags.data()),
      grid,
      isovaluesDev,
      numIsovalues,
      brickDims);

  outAabbs.reserve(n * sizeof(box3));
  auto outBegin = thrust::device_pointer_cast<box3>((box3 *)outAabbs.ptr());
  auto outEnd = thrust::copy_if(thrust::cuda::par.on(stream),
      dense.begin(),
      dense.end(),
      flags.begin(),
      outBegin,
      [] __device__(uint8_t f) { return f != uint8_t(0); });

  return size_t(outEnd - outBegin);
}

} // namespace visrtx
