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

#include "Image1D.h"

namespace visrtx {

Image1D::Image1D(DeviceGlobalState *d) : Sampler(d), m_image(this) {}

Image1D::~Image1D()
{
  cleanup();
}

void Image1D::commitParameters()
{
  Sampler::commitParameters();
  m_filter = getParamString("filter", "linear");
  m_wrap1 = getParamString("wrapMode", "clampToEdge");
  m_image = getParamObject<Array1D>("image");
}

void Image1D::finalize()
{
  cleanup();

  if (!m_image) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on image1D sampler");
    return;
  }

  ANARIDataType format = m_image->elementType();
  auto nc = numANARIChannels(format);
  if (nc == 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid texture type encountered in image1D sampler (%s)",
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
  m_texture = makeCudaTextureObject1D(cuArray, !isFp, m_filter, m_wrap1, m_borderColor);
  m_texels = makeCudaTexelObject1D(
      cuArray, !isFp, "nearest", m_wrap1, m_borderColor);

  upload();
}

bool Image1D::isValid() const
{
  return m_image;
}

int Image1D::numChannels() const
{
  ANARIDataType format = m_image->elementType();
  return numANARIChannels(format);
}

SamplerGPUData Image1D::gpuData() const
{
  SamplerGPUData retval = Sampler::gpuData();
  retval.type = SamplerType::TEXTURE1D;
  retval.image1D.texobj = m_texture;
  retval.image1D.texelTexobj = m_texels;
  retval.image1D.size = m_image->size();
  retval.image1D.invSize = 1.0f / m_image->size();

  return retval;
}

void Image1D::cleanup()
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
