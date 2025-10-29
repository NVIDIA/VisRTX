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

#include "Image2D.h"

namespace visrtx {

Image2D::Image2D(DeviceGlobalState *d) : Sampler(d), m_image(this) {}

Image2D::~Image2D()
{
  cleanup();
}

void Image2D::commitParameters()
{
  Sampler::commitParameters();
  m_filter = getParamString("filter", "linear");
  m_wrap1 = getParamString("wrapMode1", "clampToEdge");
  m_wrap2 = getParamString("wrapMode2", "clampToEdge");
  m_image = getParamObject<Array2D>("image");
}

void Image2D::finalize()
{
  cleanup();

  if (!m_image) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on image2D sampler");
    return;
  }

  const ANARIDataType format = m_image->elementType();
  auto nc = numANARIChannels(format);
  if (nc == 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid texture type encountered in image2D sampler (%s)",
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
  m_texture = makeCudaTextureObject2D(cuArray, !isFp, m_filter, m_wrap1, m_wrap2, m_borderColor);
  m_texels = makeCudaTexelObject2D(cuArray, !isFp, "nearest", m_wrap1, m_wrap2, m_borderColor);

  upload();
}

bool Image2D::isValid() const
{
  return m_image;
}

int Image2D::numChannels() const
{
  ANARIDataType format = m_image->elementType();
  return numANARIChannels(format);
}

SamplerGPUData Image2D::gpuData() const
{
  SamplerGPUData retval = Sampler::gpuData();
  retval.type = SamplerType::TEXTURE2D;
  retval.image2D.texobj = m_texture;
  retval.image2D.texelTexobj = m_texels;
  retval.image2D.size = glm::uvec2(m_image->size().x, m_image->size().y);
  retval.image2D.invSize =
      glm::vec2(1.0f / m_image->size().x, 1.0f / m_image->size().y);

  return retval;
}

void Image2D::cleanup()
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
