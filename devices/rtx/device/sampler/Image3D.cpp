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

Image3D::Image3D(DeviceGlobalState *d) : ImageSampler(d, "image3D") {}

void Image3D::commitParameters()
{
  Sampler::commitParameters();
  m_filter = getParamString("filter", "linear");
  m_wrap1 = getParamString("wrapMode1", "clampToEdge");
  m_wrap2 = getParamString("wrapMode2", "clampToEdge");
  m_wrap3 = getParamString("wrapMode3", "clampToEdge");
  setImage(getParamObject<Array3D>("image"));
}

Array3D *Image3D::image() const
{
  return static_cast<Array3D *>(imageArray());
}

void Image3D::makeTextureObjects(cudaArray_t cuArray, ANARIDataType format)
{
  const bool isFp = isFloat(format);
  // sRGB data is kept as raw bytes; the sampler does sRGB->linear in hardware.
  const bool sRGB = isSrgb8(format);
  m_texture = makeCudaTextureObject3D(
      cuArray, !isFp, m_filter, m_wrap1, m_wrap2, m_wrap3, m_borderColor, sRGB);
  m_texels = makeCudaTexelObject3D(
      cuArray, !isFp, "nearest", m_wrap1, m_wrap2, m_wrap3, m_borderColor);
}

uvec3 Image3D::imageSize() const
{
  if (!image())
    return uvec3(0);
  auto sz = image()->size();
  return uvec3(sz.x, sz.y, sz.z);
}

cudaTextureObject_t Image3D::textureObject() const
{
  return m_texture;
}

SamplerGPUData Image3D::gpuData() const
{
  SamplerGPUData retval = Sampler::gpuData();
  if (!m_texture || !m_texels)
    return retval;
  retval.type = SamplerType::TEXTURE3D;
  retval.image3D.texobj = m_texture;
  retval.image3D.texelTexobj = m_texels;
  retval.image3D.size =
      glm::uvec3(image()->size().x, image()->size().y, image()->size().z);
  retval.image3D.invSize = glm::vec3(1.0f / image()->size().x,
      1.0f / image()->size().y,
      1.0f / image()->size().z);
  return retval;
}

} // namespace visrtx
