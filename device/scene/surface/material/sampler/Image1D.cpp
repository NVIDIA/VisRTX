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
// std
#include <array>

namespace visrtx {

template <int SIZE>
using texel_t = std::array<uint8_t, SIZE>;
using texel1 = texel_t<1>;
using texel2 = texel_t<2>;
using texel3 = texel_t<3>;
using texel4 = texel_t<4>;

// Helper functions ///////////////////////////////////////////////////////////

static bool isFloat(ANARIDataType format)
{
  switch (format) {
  case ANARI_FLOAT32_VEC4:
  case ANARI_FLOAT32_VEC3:
  case ANARI_FLOAT32_VEC2:
  case ANARI_FLOAT32:
    return true;
  default:
    break;
  }
  return false;
}

static int numANARIChannels(ANARIDataType format)
{
  switch (format) {
  case ANARI_UFIXED8_RGBA_SRGB:
  case ANARI_UFIXED8_VEC4:
  case ANARI_FLOAT32_VEC4:
    return 4;
  case ANARI_UFIXED8_RGB_SRGB:
  case ANARI_UFIXED8_VEC3:
  case ANARI_FLOAT16_VEC3:
  case ANARI_FLOAT32_VEC3:
    return 3;
  case ANARI_UFIXED8_VEC2:
  case ANARI_FLOAT16_VEC2:
  case ANARI_FLOAT32_VEC2:
    return 2;
  case ANARI_FLOAT32:
    return 1;
  default:
    break;
  }
  return 0;
}

static int bytesPerChannel(ANARIDataType format)
{
  if (isFloat(format))
    return 4;
  else
    return 1;
}

static int countCudaChannels(const cudaChannelFormatDesc &desc)
{
  int channels = 0;
  if (desc.x != 0)
    channels++;
  if (desc.y != 0)
    channels++;
  if (desc.z != 0)
    channels++;
  if (desc.w != 0)
    channels++;
  return channels;
}

template <int SIZE, typename IN_VEC_T>
static texel_t<SIZE> makeTexel(IN_VEC_T v)
{
  v *= 255;
  texel_t<SIZE> retval;
  auto *in = (float *)&v;
  for (int i = 0; i < SIZE; i++)
    retval[i] = uint8_t(in[i]);
  return retval;
}

static cudaTextureAddressMode stringToAddressMode(const std::string &str)
{
  if (str == "repeat")
    return cudaAddressModeWrap;
  else if (str == "mirrorRepeat")
    return cudaAddressModeMirror;
  else
    return cudaAddressModeClamp;
}

// Image1D definitions //////////////////////////////////////////////////////

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

  std::vector<uint8_t> stagingBuffer;

  {
    stagingBuffer.resize(m_params.image->totalSize() * 4);

    if (nc == 4) {
      auto *begin = m_params.image->dataAs<vec4>();
      auto *end = begin + m_params.image->totalSize();
      std::transform(begin, end, (texel4 *)stagingBuffer.data(), [](vec4 &v) {
        return makeTexel<4>(v);
      });
    } else if (nc == 3) {
      auto *begin = m_params.image->dataAs<vec3>();
      auto *end = begin + m_params.image->totalSize();
      std::transform(begin, end, (texel4 *)stagingBuffer.data(), [](vec3 &v) {
        return makeTexel<4>(vec4(v, 1.f));
      });
    } else if (nc == 2) {
      auto *begin = m_params.image->dataAs<vec2>();
      auto *end = begin + m_params.image->totalSize();
      std::transform(begin, end, (texel2 *)stagingBuffer.data(), [](vec2 &v) {
        return makeTexel<2>(v);
      });
    } else if (nc == 1) {
      auto *begin = m_params.image->dataAs<float>();
      auto *end = begin + m_params.image->totalSize();
      std::transform(begin, end, (texel1 *)stagingBuffer.data(), [](float &v) {
        return makeTexel<1>(v);
      });
    }
  }

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
