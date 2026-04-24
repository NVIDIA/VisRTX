// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/cuda/outline.hpp"
// tsd_core
#include "tsd/core/TSDMath.hpp"
// helium
#include <helium/helium_math.h>
// thrust
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
// std
#include <algorithm>

namespace tsd::algorithms::cuda {

namespace {

__forceinline__ __device__ uint32_t shadePixel(
    uint32_t c_in, uint32_t outlineColor)
{
  auto c_in_f = helium::cvt_color_to_float4(c_in);
  auto c_h = helium::cvt_color_to_float4(outlineColor);
  auto c_out = tsd::math::lerp(c_in_f, c_h, 0.8f);
  return helium::cvt_color_to_uint32(c_out);
}

__forceinline__ __device__ bool isValidPrimitive(
    uint32_t objectId, uint32_t primitiveId)
{
  return objectId != ~0u && primitiveId != ~0u;
}

__forceinline__ __device__ uint64_t primitiveKey(
    uint32_t objectId, uint32_t primitiveId)
{
  return (uint64_t(objectId) << 32) | primitiveId;
}

} // namespace

void outlineObject(cudaStream_t stream,
    const uint32_t *objectId,
    uint32_t *color,
    uint32_t outlineId,
    uint32_t width,
    uint32_t height)
{
  thrust::for_each(thrust::cuda::par.on(stream),
      thrust::make_counting_iterator(0u),
      thrust::make_counting_iterator(width * height),
      [=] __device__(uint32_t i) {
        uint32_t y = i / width;
        uint32_t x = i % width;

        int cnt = 0;
        for (unsigned fy = std::max(0u, y - 1);
            fy <= std::min(height - 1, y + 1);
            fy++) {
          for (unsigned fx = std::max(0u, x - 1);
              fx <= std::min(width - 1, x + 1);
              fx++) {
            size_t fi = fx + size_t(width) * fy;
            if (objectId[fi] == outlineId)
              cnt++;
          }
        }

        if (cnt > 1 && cnt < 8)
          color[i] = shadePixel(
              color[i], helium::cvt_color_to_uint32({1.f, 0.5f, 0.f, 1.f}));
      });
}

void outlineObject(const uint32_t *objectId,
    uint32_t *color,
    uint32_t outlineId,
    uint32_t width,
    uint32_t height)
{
  outlineObject(cudaStream_t{0}, objectId, color, outlineId, width, height);
}

void outlinePrimitives(cudaStream_t stream,
    const uint32_t *objectId,
    const uint32_t *primitiveId,
    uint32_t *color,
    uint32_t outlineColor,
    uint32_t thickness,
    uint32_t width,
    uint32_t height)
{
  const uint32_t radius = std::max(1u, thickness);
  thrust::for_each(thrust::cuda::par.on(stream),
      thrust::make_counting_iterator(0u),
      thrust::make_counting_iterator(width * height),
      [=] __device__(uint32_t i) {
        uint32_t y = i / width;
        uint32_t x = i % width;

        bool sawValid = false;
        bool sawInvalid = false;
        bool sawDifferentValid = false;
        uint64_t firstValidKey = 0;

        for (unsigned fy = std::max(0u, y - radius);
            fy <= std::min(height - 1, y + radius) && !sawDifferentValid;
            fy++) {
          for (unsigned fx = std::max(0u, x - radius);
              fx <= std::min(width - 1, x + radius);
              fx++) {
            const size_t fi = fx + size_t(width) * fy;
            const uint32_t neighborObjectId = objectId[fi];
            const uint32_t neighborPrimitiveId = primitiveId[fi];

            if (!isValidPrimitive(neighborObjectId, neighborPrimitiveId)) {
              sawInvalid = true;
              continue;
            }

            const uint64_t key =
                primitiveKey(neighborObjectId, neighborPrimitiveId);
            if (!sawValid) {
              sawValid = true;
              firstValidKey = key;
            } else if (key != firstValidKey) {
              sawDifferentValid = true;
              break;
            }
          }
        }

        if (sawValid && (sawInvalid || sawDifferentValid))
          color[i] = shadePixel(color[i], outlineColor);
      });
}

void outlinePrimitives(const uint32_t *objectId,
    const uint32_t *primitiveId,
    uint32_t *color,
    uint32_t outlineColor,
    uint32_t thickness,
    uint32_t width,
    uint32_t height)
{
  outlinePrimitives(cudaStream_t{0},
      objectId,
      primitiveId,
      color,
      outlineColor,
      thickness,
      width,
      height);
}

} // namespace tsd::algorithms::cuda
