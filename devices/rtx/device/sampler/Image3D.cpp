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

#include "Image3D.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "optix_visrtx.h"
#include "utility/AnariTypeHelpers.h"

namespace visrtx {

Image3D::Image3D(DeviceGlobalState *d) : Sampler(d), m_image(this) {}

Image3D::~Image3D()
{
  cleanupImageTextureObjects();
  cleanupImageCudaArray();
}

void Image3D::commitParameters()
{
  Sampler::commitParameters();
  m_filter = getParamString("filter", "linear");
  m_wrap1 = getParamString("wrapMode1", "clampToEdge");
  m_wrap2 = getParamString("wrapMode2", "clampToEdge");
  m_wrap3 = getParamString("wrapMode3", "clampToEdge");
  auto *oldImage = m_image.get();
  auto *newImage = getParamObject<Array3D>("image");
  if (oldImage != newImage)
    cleanupImageCudaArray();
  m_image = newImage;
}

void Image3D::finalize()
{
  if (!m_image) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on image3D sampler");
    return;
  }

  const bool isFp = isFloat(m_image->elementType());
  cudaArray_t cuArray = m_image->acquireCUDAArray();

  cleanupImageTextureObjects();

  const ANARIDataType format = m_image->elementType();
  auto nc = numANARIChannels(format);
  if (nc == 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid texture type encountered in image3D sampler (%s)",
        anari::toString(format));
    return;
  }

  // sRGB data is kept as raw bytes; the sampler does sRGB->linear in hardware.
  const bool sRGB = isSrgb8(format);
  m_texture = makeCudaTextureObject3D(cuArray,
      !isFp,
      m_filter,
      m_wrap1,
      m_wrap2,
      m_wrap3,
      m_borderColor,
      sRGB);
  m_texels = makeCudaTexelObject3D(
      cuArray, !isFp, "nearest", m_wrap1, m_wrap2, m_wrap3, m_borderColor);

  upload();
}

bool Image3D::isValid() const
{
  return m_image;
}

uvec3 Image3D::imageSize() const
{
  if (!m_image)
    return uvec3(0);
  auto sz = m_image->size();
  return uvec3(sz.x, sz.y, sz.z);
}

int Image3D::numChannels() const
{
  ANARIDataType format = m_image->elementType();
  return numANARIChannels(format);
}

cudaTextureObject_t Image3D::textureObject() const
{
  return m_texture;
}

Array3D *Image3D::image() const
{
  return m_image.get();
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

void Image3D::cleanupImageCudaArray()
{
  if (!m_image)
    return;

  m_image->releaseCUDAArray();
}

void Image3D::cleanupImageTextureObjects()
{
  cudaDestroyTextureObject(m_texels);
  cudaDestroyTextureObject(m_texture);
  m_texels = {};
  m_texture = {};
}

} // namespace visrtx
