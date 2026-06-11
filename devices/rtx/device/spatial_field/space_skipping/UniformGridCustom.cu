/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Per-macrocell value-range grid for custom spatial fields. A plain CUDA
// kernel cannot optixDirectCall the custom SBT sampler, but the user sample
// function is an inline __device__ function reachable from any TU that includes
// the dispatch header — so this TU re-expands VISRTX_CUSTOM_SAMPLE_DISPATCH
// directly. Compiled into the library unconditionally; only active when the
// custom-field provider's CMake defines VISRTX_CUSTOM_SAMPLERS_HEADER /
// VISRTX_CUSTOM_FIELD_DATA_HEADER on the library target (else a stub).

#include <cuda_runtime_api.h>
#include <limits>
#include "UniformGrid.h"
#include "gpu/gpu_decl.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"

#ifdef VISRTX_CUSTOM_FIELD_DATA_HEADER
#include VISRTX_CUSTOM_FIELD_DATA_HEADER
#endif
#ifdef VISRTX_CUSTOM_SAMPLERS_HEADER
#include VISRTX_CUSTOM_SAMPLERS_HEADER
#endif

namespace visrtx {

#ifdef VISRTX_CUSTOM_SAMPLERS_HEADER

namespace {

constexpr int CUSTOM_VALUE_RANGE_SUPERSAMPLE = 4; // S; S^3 samples per cell (fallback)

// Only the no-dispatch supersample fallback (the kernel's #else below) samples
// the field directly; guard the helper to that case so it isn't compiled
// (and flagged unreferenced) when a value-range dispatch is provided.
#if !defined(VISRTX_CUSTOM_VALUE_RANGE_DISPATCH)                                \
    && !defined(VISRTX_CUSTOM_GLOBAL_VALUE_RANGE_DISPATCH)
VISRTX_DEVICE float sampleCustomValue(
    const CustomFieldData &data, const vec3 &P)
{
#ifdef VISRTX_CUSTOM_SAMPLE_DISPATCH
  VISRTX_CUSTOM_SAMPLE_DISPATCH(data, P)
#else
  return 0.0f;
#endif
}
#endif

#ifdef VISRTX_CUSTOM_VALUE_RANGE_DISPATCH
// Field-supplied conservative value interval over an object-space AABB. Both
// ends are real bounds; the volume cannot vanish or leak.
VISRTX_DEVICE box1 customCellValueRange(
    const CustomFieldData &data, const vec3 &boxLo, const vec3 &boxHi)
{
  VISRTX_CUSTOM_VALUE_RANGE_DISPATCH(data, boxLo, boxHi)
}
#elif defined(VISRTX_CUSTOM_GLOBAL_VALUE_RANGE_DISPATCH)
// Field-supplied conservative interval over the whole domain. Constant per cell
// (no space skipping) but never wrong.
VISRTX_DEVICE box1 customGlobalValueRange(const CustomFieldData &data)
{
  VISRTX_CUSTOM_GLOBAL_VALUE_RANGE_DISPATCH(data)
}
#endif

// Per macrocell, emit a conservative value interval over the cell's object-space
// AABB. Single pass — no lower-bound fabrication needed.
__global__ void customCellRangeGPU(box1 *valueRanges,
    ivec3 mcDims,
    box3 objectBounds,
    const SpatialFieldGPUData *sfgd,
    int S)
{
  size_t threadID = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  size_t numMCs = size_t(mcDims.x) * mcDims.y * mcDims.z;
  if (threadID >= numMCs)
    return;

  ivec3 mcID(threadID % mcDims.x,
      threadID / mcDims.x % mcDims.y,
      threadID / (size_t(mcDims.x) * mcDims.y));

  const vec3 ext = objectBounds.upper - objectBounds.lower;
  const vec3 normLo = vec3(mcID) / vec3(mcDims);
  const vec3 normHi = vec3(mcID + ivec3(1)) / vec3(mcDims);
  const vec3 objLo = objectBounds.lower + normLo * ext;
  const vec3 objHi = objectBounds.lower + normHi * ext;

  const CustomFieldData &data = sfgd->data.custom;

#ifdef VISRTX_CUSTOM_VALUE_RANGE_DISPATCH
  (void)S;
  valueRanges[threadID] = customCellValueRange(data, objLo, objHi);
#elif defined(VISRTX_CUSTOM_GLOBAL_VALUE_RANGE_DISPATCH)
  (void)S;
  valueRanges[threadID] = customGlobalValueRange(data);
#else
#warning \
    "Custom field defines samplers but no value-range dispatch; macrocell " \
    "bounds are best-effort point-supersampled and MAY mis-bound the true " \
    "field (volume can render too transparent / dark / vanish). Define " \
    "VISRTX_CUSTOM_VALUE_RANGE_DISPATCH (tight per-AABB) or " \
    "VISRTX_CUSTOM_GLOBAL_VALUE_RANGE_DISPATCH (conservative constant) for a " \
    "correctness guarantee."
  // Best-effort: point-supersample real lo AND hi (no longer clamped to 0).
  const vec3 cellExt = objHi - objLo;
  float lo = std::numeric_limits<float>::infinity();
  float hi = -std::numeric_limits<float>::infinity();
  for (int iz = 0; iz < S; ++iz)
    for (int iy = 0; iy < S; ++iy)
      for (int ix = 0; ix < S; ++ix) {
        const vec3 frac((ix + 0.5f) / S, (iy + 0.5f) / S, (iz + 0.5f) / S);
        const float v = sampleCustomValue(data, objLo + frac * cellExt);
        if (!isnan(v) && !isinf(v)) {
          lo = fminf(lo, v);
          hi = fmaxf(hi, v);
        }
      }
  valueRanges[threadID] = (lo <= hi)
      ? box1{lo, hi}
      : box1{std::numeric_limits<float>::infinity(),
             -std::numeric_limits<float>::infinity()};
#endif
}

} // namespace

void launchCustomValueRanges(box1 *valueRanges,
    ivec3 mcDims,
    box3 objectBounds,
    const SpatialFieldGPUData *dSfgd,
    cudaStream_t stream)
{
  size_t numMCs = size_t(mcDims.x) * mcDims.y * mcDims.z;
  if (numMCs == 0)
    return;
  const int threads = 256;
  const int blocks = int(iDivUp(int64_t(numMCs), threads));
  customCellRangeGPU<<<blocks, threads, 0, stream>>>(
      valueRanges, mcDims, objectBounds, dSfgd, CUSTOM_VALUE_RANGE_SUPERSAMPLE);
}

#else // no custom field configured: write empty sentinel everywhere

namespace {
__global__ void customEmptyGPU(box1 *valueRanges, size_t numMCs)
{
  size_t threadID = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (threadID >= numMCs)
    return;
  valueRanges[threadID].lower = std::numeric_limits<float>::infinity();
  valueRanges[threadID].upper = -std::numeric_limits<float>::infinity();
}
} // namespace

void launchCustomValueRanges(box1 *valueRanges,
    ivec3 mcDims,
    box3 /*objectBounds*/,
    const SpatialFieldGPUData * /*dSfgd*/,
    cudaStream_t stream)
{
  size_t numMCs = size_t(mcDims.x) * mcDims.y * mcDims.z;
  if (numMCs == 0)
    return;
  const int threads = 256;
  const int blocks = int(iDivUp(int64_t(numMCs), threads));
  customEmptyGPU<<<blocks, threads, 0, stream>>>(valueRanges, numMCs);
}

#endif

} // namespace visrtx
