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

#include "CDF.h"

#include "gpu/gpu_math.h"
#include "utility/DeviceBuffer.h"

// anari
#include <anari/frontend/anari_enums.h>

// glm
#include <glm/ext/vector_float3.hpp>
#include <glm/geometric.hpp>

// cccl
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

// cuda
#include <cuda_runtime.h>
#include <driver_types.h>
#include <texture_types.h>
#include <vector_types.h>

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(float3, ANARI_FLOAT32_VEC3);
}

namespace visrtx {

namespace {
// Importance sampling helper functions

void computeWeightedLuminance(
    const glm::vec3 *envMap, float *luminance, int width, int height)
{
  using thrust::device_pointer_cast;

  auto envMapBegin = device_pointer_cast(envMap);
  auto envMapEnd = device_pointer_cast(envMap + width * height);
  auto luminanceBegin = device_pointer_cast(luminance);

  thrust::for_each_n(
      thrust::make_counting_iterator(0), height, [=] __device__(int y) {
        // Scale distribution by the sine to get the sampling uniform. (Avoid
        // sampling more values near the poles.) See Physically Based Rendering
        // v2, chapter 14.6.5 on Infinite Area Lights, page 728.
        auto sinTheta = sinf(kPi * (y + 0.5f) / height);
        auto rowEnvMapPtr = envMapBegin + y * width;
        auto rowLuminancePtr = luminanceBegin + y * width;
        for (auto i = 0; i < width; i++) {
          glm::vec3 rgb = rowEnvMapPtr[i];
          rowLuminancePtr[i] = sinTheta * dot(rgb, {0.2126f, 0.7152f, 0.0722f});
        }
      });
}

void computeRowSums(
    const float *luminance, float *rowSums, int width, int height)
{
  using thrust::device_pointer_cast;

  auto luminancePtr = device_pointer_cast(luminance);
  auto rowSums_ptr = device_pointer_cast(rowSums);

  thrust::for_each_n(
      thrust::make_counting_iterator(0), height, [=] __device__(int y) {
        auto rowLuminancePtr = luminancePtr + y * width;
        float sum = 0.0f;
        for (int x = 0; x < width; ++x) {
          sum += rowLuminancePtr[x];
        }
        rowSums_ptr[y] = sum;
      });
}

void computeMarginalCDF(const float *rowSums, float *marginalCdf, int height)
{
  using thrust::device_pointer_cast;

  auto rowSumsBegin = device_pointer_cast(rowSums);
  auto rowSumsEnd = device_pointer_cast(rowSums + height);
  thrust::inclusive_scan(
      rowSumsBegin, rowSumsEnd, device_pointer_cast(marginalCdf));
}

void computeConditionalCDFs(
    const float *luminance, float *conditionalCdf, int width, int height)
{
  using thrust::device_pointer_cast;

  // Segmented inclusive scan: key = row index (`i / width`), so the running
  // sum resets at every row boundary. One launch family independent of
  // `height`.
  const auto keys = thrust::make_transform_iterator(
      thrust::counting_iterator<int>(0),
      [width] __host__ __device__(int i) { return i / width; });

  thrust::inclusive_scan_by_key(keys,
      keys + width * height,
      device_pointer_cast(luminance),
      device_pointer_cast(conditionalCdf));
}

void normalizeCDF(thrust::device_ptr<float> cdf, int n)
{
  if (n <= 0)
    return;
  const float total = cdf[n - 1];
  if (total > 0.0f) {
    thrust::transform(
        cdf, cdf + n, cdf, [total] __device__(float x) { return x / total; });
  } else {
    // Empty distribution; fill with uniform values so sampling doesn't walk off
    // the end.
    thrust::fill(cdf, cdf + n, 1.0f);
  }
}

void normalizeMarginalCDF(float *marginalCdf, int height)
{
  normalizeCDF(thrust::device_pointer_cast(marginalCdf), height);
}

__global__ void normalizeConditionalCDFsKernel(float *cdf, int width)
{
  // One block per row. Read the row total (cdf[width-1] = sum after the
  // inclusive scan) into shared memory, then normalize each element in
  // parallel. Empty rows (total ≤ 0) fill with 1.0 so a downstream sampler
  // walks the row uniformly instead of running off the end.
  const int y = blockIdx.x;
  const int tid = threadIdx.x;
  float *row = cdf + y * width;

  __shared__ float s_total;
  if (tid == 0)
    s_total = row[width - 1];
  __syncthreads();

  const bool empty = !(s_total > 0.0f);
  const float invTotal = empty ? 0.0f : 1.0f / s_total;

  for (int x = tid; x < width; x += blockDim.x)
    row[x] = empty ? 1.0f : row[x] * invTotal;
}

void normalizeConditionalCDFs(float *d_conditional_cdf, int width, int height)
{
  // One block per row, all `height` rows in parallel.
  if (width <= 0 || height <= 0)
    return;
  constexpr int kThreadsPerBlock = 256;
  normalizeConditionalCDFsKernel<<<height, kThreadsPerBlock>>>(
      d_conditional_cdf, width);
}

} // namespace

float generateCDFTables(const float *luminanceImage,
    int width,
    int height,
    DeviceBuffer *marginalCdf,
    DeviceBuffer *conditionalCdf)
{
  using thrust::device_pointer_cast;

  DeviceBuffer rowSums;

  rowSums.reserve(height * sizeof(float));
  marginalCdf->reserve(height * sizeof(float));
  conditionalCdf->reserve(width * height * sizeof(float));

  computeRowSums(luminanceImage, rowSums.ptrAs<float>(), width, height);
  computeMarginalCDF(
      rowSums.ptrAs<const float>(), marginalCdf->ptrAs<float>(), height);
  computeConditionalCDFs(
      luminanceImage, conditionalCdf->ptrAs<float>(), width, height);

  // Compute pdfWeight

  // Not the best, but accumulation operations of cdfs accumulate error.
  // Lets recompute the total luminance from the luminance array
  // to avoid this.
  auto totalLuminance = reduce(device_pointer_cast(luminanceImage),
      device_pointer_cast(luminanceImage) + width * height);

  // Equirectangular Jacobian |dω/d(u,v)| = 2π²·sinθ; the sinθ weighting is
  // already folded into the CDF luminance, so the per-pixel area factor is
  // 2π²/(W·H) and pdf_ω = (L/totalL) · (W·H)/(2π²).
  // A zero-luminance map produces an inf weight; return 0 instead.
  const float equirectJacobian = 2.0f * kPi * kPi / (width * height);
  const float weight =
      totalLuminance > 0.0f ? 1.0f / (totalLuminance * equirectJacobian) : 0.0f;

  // Normalize both tables
  normalizeMarginalCDF(marginalCdf->ptrAs<float>(), height);
  normalizeConditionalCDFs(conditionalCdf->ptrAs<float>(), width, height);

  return weight;
}

float generateCDFTables(const glm::vec3 *rgbImage,
    int width,
    int height,
    DeviceBuffer *marginalCdf,
    DeviceBuffer *conditionalCdf)
{
  using thrust::device_pointer_cast;

  DeviceBuffer luminance;
  DeviceBuffer rowSums;

  luminance.reserve(width * height * sizeof(float));

  computeWeightedLuminance(rgbImage, luminance.ptrAs<float>(), width, height);

  return generateCDFTables(luminance.ptrAs<const float>(),
      width,
      height,
      marginalCdf,
      conditionalCdf);
}

} // namespace visrtx
