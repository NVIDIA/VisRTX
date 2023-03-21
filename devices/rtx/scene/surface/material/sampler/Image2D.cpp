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
#include "ImageSamplerHelpers.h"

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

  m_params.filter = getParamString("filter", "nearest");
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

  // Create CUDA texture //

  std::vector<uint8_t> stagingBuffer(image.totalSize() * 4);

  if (nc == 4) {
    if (isFloat(format))
      transformToStagingBuffer<4, float>(image, stagingBuffer.data());
    else if (isFixed32(format))
      transformToStagingBuffer<4, uint32_t>(image, stagingBuffer.data());
    else if (isFixed16(format))
      transformToStagingBuffer<4, uint16_t>(image, stagingBuffer.data());
    else if (isFixed8(format))
      transformToStagingBuffer<4, uint8_t>(image, stagingBuffer.data());
    else if (isSrgb8(format))
      transformToStagingBuffer<4, uint8_t, true>(image, stagingBuffer.data());
  } else if (nc == 3) {
    if (isFloat(format))
      transformToStagingBuffer<3, float>(image, stagingBuffer.data());
    else if (isFixed32(format))
      transformToStagingBuffer<3, uint32_t>(image, stagingBuffer.data());
    else if (isFixed16(format))
      transformToStagingBuffer<3, uint16_t>(image, stagingBuffer.data());
    else if (isFixed8(format))
      transformToStagingBuffer<3, uint8_t>(image, stagingBuffer.data());
    else if (isSrgb8(format))
      transformToStagingBuffer<3, uint8_t, true>(image, stagingBuffer.data());
  } else if (nc == 2) {
    if (isFloat(format))
      transformToStagingBuffer<2, float>(image, stagingBuffer.data());
    else if (isFixed32(format))
      transformToStagingBuffer<2, uint32_t>(image, stagingBuffer.data());
    else if (isFixed16(format))
      transformToStagingBuffer<2, uint16_t>(image, stagingBuffer.data());
    else if (isFixed8(format))
      transformToStagingBuffer<2, uint8_t>(image, stagingBuffer.data());
    else if (isSrgb8(format))
      transformToStagingBuffer<2, uint8_t, true>(image, stagingBuffer.data());
  } else if (nc == 1) {
    if (isFloat(format))
      transformToStagingBuffer<1, float>(image, stagingBuffer.data());
    else if (isFixed32(format))
      transformToStagingBuffer<1, uint32_t>(image, stagingBuffer.data());
    else if (isFixed16(format))
      transformToStagingBuffer<1, uint16_t>(image, stagingBuffer.data());
    else if (isFixed8(format))
      transformToStagingBuffer<1, uint8_t>(image, stagingBuffer.data());
    else if (isSrgb8(format))
      transformToStagingBuffer<1, uint8_t, true>(image, stagingBuffer.data());
  }

  if (nc == 3)
    nc = 4;

  auto desc = cudaCreateChannelDesc(nc >= 1 ? 8 : 0,
      nc >= 2 ? 8 : 0,
      nc >= 3 ? 8 : 0,
      nc >= 3 ? 8 : 0,
      cudaChannelFormatKindUnsigned);

  const auto size = image.size();
  cudaMallocArray(&m_cudaArray, &desc, size.x, size.y);
  cudaMemcpy2DToArray(m_cudaArray,
      0,
      0,
      stagingBuffer.data(),
      size.x * nc,
      size.x * nc,
      size.y,
      cudaMemcpyHostToDevice);

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = m_cudaArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = stringToAddressMode(m_params.wrap1);
  texDesc.addressMode[1] = stringToAddressMode(m_params.wrap2);
  texDesc.filterMode =
      m_params.filter == "nearest" ? cudaFilterModePoint : cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords = 1;

  cudaCreateTextureObject(&m_textureObject, &resDesc, &texDesc, nullptr);

  upload();
}

SamplerGPUData Image2D::gpuData() const
{
  SamplerGPUData retval = Sampler::gpuData();
  retval.type = SamplerType::TEXTURE2D;
  retval.image2D.texobj = m_textureObject;
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
