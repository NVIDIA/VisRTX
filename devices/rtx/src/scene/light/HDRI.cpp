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
#include "sampling/CDF.h"
#include "utility/DeviceBuffer.h"
#include "utility/CudaImageTexture.h"

// anari
#include <anari/frontend/anari_enums.h>

// glm
#include <glm/ext/matrix_float3x3.hpp>
#include <glm/ext/vector_float3.hpp>

// cuda
#include <cuda_runtime.h>

// std
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

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
  m_radiance = nullptr;
  if (auto radiance = getParamObject<Array2D>("radiance")) {
    if (radiance->elementType() == ANARI_FLOAT32_VEC3) {
      m_radiance = getParamObject<Array2D>("radiance");
    } else {
      reportMessage(ANARI_SEVERITY_WARNING,
          "invalid element type %s for 'radiance' on HDRI light", anari::toString(radiance->elementType()));
    }
  }
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
  retval.hdri.xfm = transpose(mat3(right, forward, up));
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

} // namespace visrtx
