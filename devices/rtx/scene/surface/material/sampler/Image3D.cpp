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

#include "Image3D.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "optix_visrtx.h"
#include "utility/AnariTypeHelpers.h"

namespace visrtx {

Image3D::Image3D(DeviceGlobalState *d) : Sampler(d), m_image(this) {}

Image3D::~Image3D()
{
  cleanup();
}

void Image3D::commitParameters()
{
  Sampler::commitParameters();
  m_filter = getParamString("filter", "linear");
  m_wrap1 = getParamString("wrapMode1", "clampToEdge");
  m_wrap2 = getParamString("wrapMode2", "clampToEdge");
  m_wrap3 = getParamString("wrapMode3", "clampToEdge");
  m_image = getParamObject<Array3D>("image");
}

void Image3D::finalize()
{
  cleanup();

  if (!m_image) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on image3D sampler");
    return;
  }

  const ANARIDataType format = m_image->elementType();
  auto nc = numANARIChannels(format);
  if (nc == 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid texture type encountered in image3D sampler (%s)",
        anari::toString(format));
    return;
  }

  cudaArray_t cuArray = {};
  bool isFp = isFloat(m_image->elementType());
  if (isFp) {
    cuArray = m_image->acquireCUDAArrayFloat();
  } else {
    cuArray = m_image->acquireCUDAArrayUint8();
  }
  m_texture = makeCudaTextureObject(
      cuArray, !isFp, m_filter, m_wrap1, m_wrap2, m_wrap3);
  m_texels = makeCudaTextureObject(
      cuArray, !isFp, "nearest", m_wrap1, m_wrap2, m_wrap3, false);

  upload();
}

bool Image3D::isValid() const
{
  return m_image;
}

int Image3D::numChannels() const
{
  ANARIDataType format = m_image->elementType();
  return numANARIChannels(format);
}

SamplerGPUData Image3D::gpuData() const
{
  SamplerGPUData retval = Sampler::gpuData();
  retval.type = SamplerType::TEXTURE3D;
  retval.image3D.texobj = m_texture;
  retval.image3D.texelTexobj = m_texels;
  retval.image3D.size =
      glm::uvec3(m_image->size().x, m_image->size().y, m_image->size().z);
  retval.image3D.invSize = glm::vec3(1.0f / m_image->size().x,
      1.0f / m_image->size().y,
      1.0f / m_image->size().z);
  return retval;
}

void Image3D::cleanup()
{
  if (m_image && m_texture) {
    cudaDestroyTextureObject(m_texels);
    cudaDestroyTextureObject(m_texture);
    if (isFloat(m_image->elementType())) {
      m_image->releaseCUDAArrayFloat();
    } else {
      m_image->releaseCUDAArrayUint8();
    }
  }
}

} // namespace visrtx
