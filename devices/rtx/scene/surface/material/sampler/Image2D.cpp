/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Image2D::Image2D(DeviceGlobalState *d) : Sampler(d) {}

Image2D::~Image2D()
{
  cleanup();
}

void Image2D::commit()
{
  Sampler::commit();

  cleanup();

  m_params.filter = getParamString("filter", "linear");
  m_params.wrap1 = getParamString("wrapMode1", "clampToEdge");
  m_params.wrap2 = getParamString("wrapMode2", "clampToEdge");
  m_params.image = getParamObject<Array2D>("image");

  if (!m_params.image) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on image2D sampler");
    return;
  }

  const ANARIDataType format = m_params.image->elementType();
  auto nc = numANARIChannels(format);
  if (nc == 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid texture type encountered in image2D sampler (%s)",
        anari::toString(format));
    return;
  }

  auto &image = *m_params.image;
  image.addCommitObserver(this);

  auto cuArray = image.acquireCUDAArrayUint8();
  m_texture = makeCudaTextureObject(
      cuArray, true, m_params.filter, m_params.wrap1, m_params.wrap2);

  upload();
}

SamplerGPUData Image2D::gpuData() const
{
  SamplerGPUData retval = Sampler::gpuData();
  retval.type = SamplerType::TEXTURE2D;
  retval.image2D.texobj = m_texture;
  return retval;
}

int Image2D::numChannels() const
{
  ANARIDataType format = m_params.image->elementType();
  return numANARIChannels(format);
}

bool Image2D::isValid() const
{
  return m_params.image;
}

void Image2D::cleanup()
{
  if (m_params.image) {
    if (m_texture) {
      cudaDestroyTextureObject(m_texture);
      m_params.image->releaseCUDAArrayUint8();
    }
    m_params.image->removeCommitObserver(this);
  }
}

} // namespace visrtx
