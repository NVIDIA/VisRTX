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

#include "CompressedImage2D.h"
#include <anari/frontend/anari_enums.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "utility/CudaImageTexture.h"

// Texture size is specified as a uint64_2 in the specifications.
// Let's make sure we can process that.
namespace anari {
  ANARI_TYPEFOR_SPECIALIZATION(glm::u64vec2, ANARI_UINT64_VEC2);
}


namespace visrtx {

CompressedImage2D::CompressedImage2D(DeviceGlobalState *d)
    : Sampler(d), m_image(this)
{}

CompressedImage2D::~CompressedImage2D()
{
  cleanup();
}

void CompressedImage2D::commitParameters()
{
  Sampler::commitParameters();
  m_filter = getParamString("filter", "linear");
  m_wrap1 = getParamString("wrapMode1", "clampToEdge");
  m_wrap2 = getParamString("wrapMode2", "clampToEdge");
  m_image = getParamObject<Array1D>("image");
  auto size64 = getParam("size", glm::u64vec2(0, 0));
  m_size = {size64.x, size64.y};
  m_format = getParamString("format", "");
}

void CompressedImage2D::finalize()
{
  cleanup();

  if (!m_image) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on CompressedImage2D sampler");
    return;
  }

  const ANARIDataType format = m_image->elementType();
  m_cuArray = {};
  if (format != ANARI_UINT8 && format != ANARI_INT8) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid texture type encountered in CompressedImage2D sampler (%s). Must be ANARI_UINT8 or ANARI_INT8.",
        anari::toString(format));
    return;
  }

  cudaChannelFormatKind channelFormatKind;

  if (m_format == "BC1_RGB" || m_format == "BC1_RGBA") {
    channelFormatKind = cudaChannelFormatKindUnsignedBlockCompressed1;
  } else if (m_format == "BC1_RGB_SRGB" || m_format == "BC1_RGBA_SRGB") {
    channelFormatKind = cudaChannelFormatKindUnsignedBlockCompressed1SRGB;
  } else if (m_format == "BC2") {
    channelFormatKind = cudaChannelFormatKindUnsignedBlockCompressed2;
  } else if (m_format == "BC2_SRGB") {
    channelFormatKind = cudaChannelFormatKindUnsignedBlockCompressed2SRGB;
  } else if (m_format == "BC3") {
    channelFormatKind = cudaChannelFormatKindUnsignedBlockCompressed3;
  } else if (m_format == "BC3_SRGB") {
    channelFormatKind = cudaChannelFormatKindUnsignedBlockCompressed3SRGB;
  } else if (m_format == "BC4") {
    channelFormatKind = cudaChannelFormatKindUnsignedBlockCompressed4;
  } else if (m_format == "BC4_SNORM") {
    channelFormatKind = cudaChannelFormatKindSignedBlockCompressed4;
  } else if (m_format == "BC5") {
    channelFormatKind = cudaChannelFormatKindUnsignedBlockCompressed5;
  } else if (m_format == "BC5_SNORM") {
    channelFormatKind = cudaChannelFormatKindUnsignedBlockCompressed5;
  } else if (m_format == "BC6H_UFLOAT") {
    channelFormatKind = cudaChannelFormatKindUnsignedBlockCompressed6H;
  } else if (m_format == "BC6H_SFLOAT") {
    channelFormatKind = cudaChannelFormatKindSignedBlockCompressed6H;
  } else if (m_format == "BC7") {
    channelFormatKind = cudaChannelFormatKindUnsignedBlockCompressed7;
  } else if (m_format == "BC7_SRGB") {
    channelFormatKind = cudaChannelFormatKindUnsignedBlockCompressed7SRGB;
  } else {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid texture format encountered in CompressedImage2D sampler (%s)",
        m_format);
    return;
  }
  makeCudaCompressedTextureArray(
      m_cuArray, m_size, *m_image.get(), channelFormatKind);

  m_texture = makeCudaCompressedTextureObject(m_cuArray,
      m_filter,
      m_wrap1,
      m_wrap2,
      "clampToEdge",
      true,
      channelFormatKind);
  m_texels = makeCudaCompressedTextureObject(m_cuArray,
      "nearest",
      m_wrap1,
      m_wrap2,
      "clampToEdge",
      false,
      channelFormatKind);

  upload();
}

bool CompressedImage2D::isValid() const
{
  return m_image;
}

int CompressedImage2D::numChannels() const
{
  if (m_format == "BC1_RGB" || m_format == "BC1_RGBA") {
    return 4;
  } else if (m_format == "BC1_RGB_SRGB" || m_format == "BC1_RGBA_SRGB") {
    return 4;
  } else if (m_format == "BC2" || m_format == "BC2_SRGB") {
    return 4;
  } else if (m_format == "BC3" || m_format == "BC3_SRGB") {
    return 4;
  } else if (m_format == "BC4" || m_format == "BC4_SNORM") {
    return 1;
  } else if (m_format == "BC5" || m_format == "BC5_SNORM") {
    return 2;
  } else if (m_format == "BC6H_UFLOAT" || m_format == "BC6H_SFLOAT") {
    return 4;
  } else if (m_format == "BC7" || m_format == "BC7_SRGB") {
    return 4;
  } else {
    return 0;
  }
}

SamplerGPUData CompressedImage2D::gpuData() const
{
  SamplerGPUData retval = Sampler::gpuData();
  retval.type = SamplerType::TEXTURE2D;
  retval.image2D.texobj = m_texture;
  retval.image2D.texelTexobj = m_texels;
  retval.image2D.size = m_size;
  retval.image2D.invSize =
      glm::vec2(1.0f / m_size.x, 1.0f / m_size.y);

  return retval;
}

void CompressedImage2D::cleanup()
{
  if (m_image && m_texture) {
    cudaDestroyTextureObject(m_texels);
    cudaDestroyTextureObject(m_texture);
    cudaFreeArray(m_cuArray);
  }
}

} // namespace visrtx
