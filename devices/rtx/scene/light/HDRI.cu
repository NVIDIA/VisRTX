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

#include "HDRI.h"

#include "array/Array.h"
#include "utility/DeviceBuffer.h"
#ifdef VISRTX_ENABLE_HDRI_SAMPLING_DEBUG
#include "utility/PPM.h"
#endif

#include <anari/anari_cpp/Traits.h>
#include <anari/frontend/anari_enums.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <sys/types.h>
#include <texture_types.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <vector_types.h>
#include <cstdint>
#include <cub/thread/thread_operators.cuh>
#include <fstream>
#include <glm/matrix.hpp>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <algorithm>
#include <cstdio>
#include <cub/thread/thread_operators.cuh>
#include <glm/ext/vector_float3.hpp>
#include <glm/gtc/color_space.hpp>
#include <glm/gtx/color_space.hpp>
#include <iostream>
#include <iterator>
#include <vector>
#include "utility/DeviceBuffer.h"

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(float3, ANARI_FLOAT32_VEC3);
}

namespace visrtx {

HDRI::HDRI(DeviceGlobalState *d) : Light(d), m_radiance(this) {}

HDRI::~HDRI()
{
  cleanup();
}

void HDRI::commitParameters()
{
  Light::commitParameters();
  m_radiance = getParamObject<Array2D>("radiance");
}

void HDRI::finalize()
{
  cleanup();

  if (!m_radiance) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'radiance' on HDRI light");
    return;
  }

  m_direction = getParam<vec3>("direction", vec3(1.f, 0.f, 0.f));
  m_up = getParam<vec3>("up", vec3(0.f, 0.f, 1.f));
  m_scale = getParam<float>("scale", 1.f);
  m_visible = getParam<bool>("visible", true);

  cudaArray_t cuArray = {};
  const bool isFp = isFloat(m_radiance->elementType());
  if (isFp)
    cuArray = m_radiance->acquireCUDAArrayFloat();
  else
    cuArray = m_radiance->acquireCUDAArrayUint8();

  m_size = {m_radiance->size(0), m_radiance->size(1)};

  m_pdfWeight =
      generateCDFTables(m_radiance->dataAs<glm::vec3>(AddressSpace::GPU),
          m_radiance->size(0),
          m_radiance->size(1),
          &m_marginalCDF,
          &m_conditionalCDF);

  m_radianceTex =
      makeCudaTextureObject(cuArray, !isFp, "linear", "repeat", "clampToEdge");

#ifdef VISRTX_ENABLE_HDRI_SAMPLING_DEBUG
  fprintf(stderr, "HDRI light setup: %p\n", this);
  cudaMalloc(&m_samples, m_size.x * m_size.y * sizeof(unsigned int));
  cudaMemset(m_samples, 0, m_size.x * m_size.y * sizeof(unsigned int));
#endif
  upload();
}

bool HDRI::isValid() const
{
  return m_radiance;
}

bool HDRI::isHDRI() const
{
  return true;
}

LightGPUData HDRI::gpuData() const
{
  auto retval = Light::gpuData();

  const vec3 up = -glm::normalize(m_up);
  const vec3 forward = -glm::normalize(glm::cross(up, m_direction));
  const vec3 right = glm::normalize(glm::cross(forward, up));

  retval.type = LightType::HDRI;
  // The matrix is orthogonal, so we can use the transpose as the inverse
  retval.hdri.xfm = glm::transpose(mat3(right, forward, up));
  retval.hdri.scale = m_scale;
  retval.hdri.size = m_size;
  retval.hdri.radiance = m_radianceTex;
  retval.hdri.visible = m_visible;
  retval.hdri.marginalCDF = m_marginalCDF.ptrAs<const float>();
  retval.hdri.conditionalCDF = m_conditionalCDF.ptrAs<const float>();
  retval.hdri.pdfWeight = m_pdfWeight;
#ifdef VISRTX_ENABLE_HDRI_SAMPLING_DEBUG
  retval.hdri.samples = m_samples;
#endif
  return retval;
}

void HDRI::cleanup()
{
  if (m_radiance && m_radianceTex) {
    cudaDestroyTextureObject(m_radianceTex);
    if (isFloat(m_radiance->elementType()))
      m_radiance->releaseCUDAArrayFloat();
    else
      m_radiance->releaseCUDAArrayUint8();
  }

#ifdef VISRTX_ENABLE_HDRI_SAMPLING_DEBUG
  if (m_samples) {
    fprintf(stderr, "Writing HDRI sampling debug data to file...\n");
    std::vector<unsigned int> sampleData(m_size.x * m_size.y);
    cudaMemcpy(sampleData.data(),
        m_samples,
        m_size.x * m_size.y * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);

    static unsigned int counter = 0;
    auto filename =
        std::string("hdri_samples_") + std::to_string(counter++) + ".pfm";

    std::vector<float> sampleDataF(m_size.x * m_size.y);
    std::copy(sampleData.begin(), sampleData.end(), sampleDataF.begin());
    auto maxSample = -1.0f;

    std::ofstream out(filename, std::ios::out | std::ios::binary);
    if (out.is_open()) {
      out << "Pf\n" << m_size.x << " " << m_size.y << "\n" << maxSample << "\n";
      out.write(reinterpret_cast<const char *>(sampleDataF.data()),
          m_size.x * m_size.y * sizeof(float));
      out.close();
    } else {
      fprintf(stderr, "Failed to open file for writing HDRI samples.\n");
    }

    cudaFree(m_samples);
    m_samples = nullptr;
  }
#endif
}

namespace {
// Importance sampling helper functions

void computeWeightedLuminance(
    const glm::vec3 *envMap, float *luminance, int width, int height)
{
  auto envMapBegin = thrust::device_pointer_cast(envMap);
  auto envMapEnd = thrust::device_pointer_cast(envMap + width * height);
  auto luminanceBegin = thrust::device_pointer_cast(luminance);

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
  thrust::device_ptr<const float> luminancePtr(luminance);
  thrust::device_ptr<float> rowSums_ptr(rowSums);

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
  auto cdf= device_pointer_cast(marginalCdf);
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

float HDRI::generateCDFTables(const glm::vec3 *envMap,
    int width,
    int height,
    DeviceBuffer *marginalCdf,
    DeviceBuffer *conditionalCdf)
{
  DeviceBuffer luminance;
  DeviceBuffer rowSums;

  luminance.reserve(width * height * sizeof(float));
  rowSums.reserve(height * sizeof(float));
  marginalCdf->reserve(height * sizeof(float));
  conditionalCdf->reserve(width * height * sizeof(float));

  computeWeightedLuminance(envMap, luminance.ptrAs<float>(), width, height);
  computeRowSums(
      luminance.ptrAs<const float>(), rowSums.ptrAs<float>(), width, height);
  computeMarginalCDF(
      rowSums.ptrAs<const float>(), marginalCdf->ptrAs<float>(), height);
  computeConditionalCDFs(luminance.ptrAs<const float>(),
      conditionalCdf->ptrAs<float>(),
      width,
      height);

#ifdef VISRTX_ENABLE_HDRI_SAMPLING_DEBUG
  saveToPfm("luminance.pfm", luminance.ptrAs<const float>(), width, height);
  saveToPfm("row_sums.pfm", rowSums.ptrAs<const float>(), height, 1);
  saveToPfm("marginal_cdf.pfm", marginalCdf->ptrAs<const float>(), height, 1);
  saveToPfm("conditional_cdf.pfm",
      conditionalCdf->ptrAs<const float>(),
      width,
      height);
#endif

  // Compute pdfWeight
  using thrust::device_pointer_cast;
  // Not the best, but accumulation operations of cdfs accumulate error.
  // Lets recompute the total luminance from the luminance array
  // to avoid this.
  auto totalLuminance = reduce(
      device_pointer_cast(luminance.ptrAs<const float>()),
          device_pointer_cast(luminance.ptrAs<const float>()) + width * height);

  float angularArea = 4.0f * float(M_PI) / (width * height);
  float weight = 1.0f / (totalLuminance * angularArea);

  // Normalize both tables
  normalizeMarginalCDF(marginalCdf->ptrAs<float>(), height);
  normalizeConditionalCDFs(conditionalCdf->ptrAs<float>(), width, height);

  return weight;
}

} // namespace visrtx
