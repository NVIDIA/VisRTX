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

#include "CDF.h"

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
        auto sinTheta = sinf(float(M_PI) * (y + 0.5f) / height);
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

  for (int y = 0; y < height; ++y) {
    auto luminanceRow = device_pointer_cast(luminance + y * width);
    auto conditionalCdfRow = device_pointer_cast(conditionalCdf + y * width);
    thrust::inclusive_scan(
        luminanceRow, luminanceRow + width, conditionalCdfRow);
  }
}

void normalizeMarginalCDF(float *marginalCdf, int height)
{
  using thrust::device_pointer_cast;

  auto cdf = device_pointer_cast(marginalCdf);
  thrust::transform(cdf,
      cdf + height,
      cdf,
      [total = cdf[height - 1]] __device__(float x) { return x / total; });
}

void normalizeConditionalCDFs(float *d_conditional_cdf, int width, int height)
{
  using thrust::device_pointer_cast;

  for (int y = 0; y < height; ++y) {
    auto cdfRow = device_pointer_cast(d_conditional_cdf + y * width);
    thrust::transform(
        cdfRow, cdfRow + width, cdfRow, [total = cdfRow[width - 1]] __device__(float x) {
          return x / total;
        });
  }
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
  computeConditionalCDFs(luminanceImage,
      conditionalCdf->ptrAs<float>(),
      width,
      height);
  
  // Compute pdfWeight
  
  // Not the best, but accumulation operations of cdfs accumulate error.
  // Lets recompute the total luminance from the luminance array
  // to avoid this.
  auto totalLuminance = reduce(
      device_pointer_cast(luminanceImage),
          device_pointer_cast(luminanceImage) + width * height);

  float angularArea = 4.0f * float(M_PI) / (width * height);
  float weight = 1.0f / (totalLuminance * angularArea);

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

  return generateCDFTables(
      luminance.ptrAs<const float>(), width, height, marginalCdf, conditionalCdf);
}

} // namespace visrtx
