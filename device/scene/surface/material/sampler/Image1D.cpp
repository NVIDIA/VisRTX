/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ImageSamplerHelpers.h"

namespace visrtx {

Image1D::~Image1D()
{
  cleanup();
}

void Image1D::commit()
{
  Sampler::commit();

  cleanup();

  m_params.filter = getParam<std::string>("filter", "nearest");
  m_params.wrap1 = getParam<std::string>("wrapMode1", "clampToEdge");
  m_params.image = getParamObject<Array1D>("image");

  if (!m_params.image) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on image1D sampler");
    return;
  }

  ANARIDataType format = m_params.image->elementType();
  if (!isFloat(format)) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "only 32-bit float data is supported for image1D samplers");
    return;
  }

  auto nc = numANARIChannels(format);
  if (nc == 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid texture type encountered in image1D sampler");
    return;
  }

  m_params.image->addCommitObserver(this);

  // Create CUDA texture //

  const auto size = m_params.image->size();

  std::vector<uint8_t> stagingBuffer(m_params.image->totalSize() * 4);

  if (nc == 4)
    transformToStagingBuffer<4, vec4>(*m_params.image, stagingBuffer.data());
  else if (nc == 3)
    transformToStagingBuffer<3, vec3>(*m_params.image, stagingBuffer.data());
  else if (nc == 2)
    transformToStagingBuffer<2, vec2>(*m_params.image, stagingBuffer.data());
  else if (nc == 1)
    transformToStagingBuffer<1, float>(*m_params.image, stagingBuffer.data());

  if (nc == 3)
    nc = 4;

  auto desc = cudaCreateChannelDesc(nc >= 1 ? 8 : 0,
      nc >= 2 ? 8 : 0,
      nc >= 3 ? 8 : 0,
      nc >= 3 ? 8 : 0,
      cudaChannelFormatKindUnsigned);

  cudaMallocArray(&m_cudaArray, &desc, size, 1);
  cudaMemcpy2DToArray(m_cudaArray,
      0,
      0,
      stagingBuffer.data(),
      size * nc,
      size * nc,
      1,
      cudaMemcpyHostToDevice);

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = m_cudaArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = stringToAddressMode(m_params.wrap1);
  texDesc.filterMode =
      m_params.filter == "nearest" ? cudaFilterModePoint : cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords = 1;

  cudaCreateTextureObject(&m_textureObject, &resDesc, &texDesc, nullptr);
}

SamplerGPUData Image1D::gpuData() const
{
  SamplerGPUData retval = Sampler::gpuData();
  retval.type = SamplerType::TEXTURE1D;
  retval.image1D.texobj = m_textureObject;
  return retval;
}

int Image1D::numChannels() const
{
  ANARIDataType format = m_params.image->elementType();
  return numANARIChannels(format);
}

void Image1D::cleanup()
{
  if (m_textureObject)
    cudaDestroyTextureObject(m_textureObject);
  if (m_cudaArray)
    cudaFreeArray(m_cudaArray);
  m_textureObject = {};
  m_cudaArray = {};
  if (m_params.image)
    m_params.image->removeCommitObserver(this);
}

} // namespace visrtx
