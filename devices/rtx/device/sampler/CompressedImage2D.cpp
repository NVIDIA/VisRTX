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

#include "CompressedImage2D.h"

#include <anari/frontend/anari_enums.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <optional>
#include <string_view>

// Texture size is specified as a uint64_2 in the specifications.
// Let's make sure we can process that.
namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(glm::u64vec2, ANARI_UINT64_VEC2);
}

namespace visrtx {

namespace {

struct CompressedFormat
{
  cudaChannelFormatKind kind;
  int numChannels;
};

std::optional<CompressedFormat> parseFormat(std::string_view format)
{
  if (format == "BC1_RGB" || format == "BC1_RGBA")
    return CompressedFormat{cudaChannelFormatKindUnsignedBlockCompressed1, 4};
  if (format == "BC1_RGB_SRGB" || format == "BC1_RGBA_SRGB")
    return CompressedFormat{
        cudaChannelFormatKindUnsignedBlockCompressed1SRGB, 4};
  if (format == "BC2")
    return CompressedFormat{cudaChannelFormatKindUnsignedBlockCompressed2, 4};
  if (format == "BC2_SRGB")
    return CompressedFormat{
        cudaChannelFormatKindUnsignedBlockCompressed2SRGB, 4};
  if (format == "BC3")
    return CompressedFormat{cudaChannelFormatKindUnsignedBlockCompressed3, 4};
  if (format == "BC3_SRGB")
    return CompressedFormat{
        cudaChannelFormatKindUnsignedBlockCompressed3SRGB, 4};
  if (format == "BC4")
    return CompressedFormat{cudaChannelFormatKindUnsignedBlockCompressed4, 1};
  if (format == "BC4_SNORM")
    return CompressedFormat{cudaChannelFormatKindSignedBlockCompressed4, 1};
  if (format == "BC5")
    return CompressedFormat{cudaChannelFormatKindUnsignedBlockCompressed5, 2};
  if (format == "BC5_SNORM")
    return CompressedFormat{cudaChannelFormatKindSignedBlockCompressed5, 2};
  if (format == "BC6H_UFLOAT")
    return CompressedFormat{cudaChannelFormatKindUnsignedBlockCompressed6H, 4};
  if (format == "BC6H_SFLOAT")
    return CompressedFormat{cudaChannelFormatKindSignedBlockCompressed6H, 4};
  if (format == "BC7")
    return CompressedFormat{cudaChannelFormatKindUnsignedBlockCompressed7, 4};
  if (format == "BC7_SRGB")
    return CompressedFormat{
        cudaChannelFormatKindUnsignedBlockCompressed7SRGB, 4};
  return std::nullopt;
}

} // namespace

CompressedImage2D::CompressedImage2D(DeviceGlobalState *d)
    : Sampler(d), m_image(this)
{}

CompressedImage2D::~CompressedImage2D()
{
  cleanupTextureObjects();
  cleanupCudaArray();
}

void CompressedImage2D::commitParameters()
{
  Sampler::commitParameters();
  m_filter = getParamString("filter", "linear");
  m_wrap1 = getParamString("wrapMode1", "clampToEdge");
  m_wrap2 = getParamString("wrapMode2", "clampToEdge");
  const auto size64 = getParam("size", glm::u64vec2(0, 0));
  m_size = {uint32_t(size64.x), uint32_t(size64.y)};
  if (size64.x > UINT32_MAX || size64.y > UINT32_MAX)
    m_size = {};
  m_format = getParamString("format", "");
  m_image = getParamObject<Array1D>("image");
}

void CompressedImage2D::finalize()
{
  // Texture objects carry filter/wrap/border state: always rebuild them.
  cleanupTextureObjects();

  if (!m_image) {
    cleanupCudaArray();
    upload();
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on compressedImage2D sampler");
    return;
  }

  const ANARIDataType elementType = m_image->elementType();
  if (elementType != ANARI_UINT8 && elementType != ANARI_INT8) {
    cleanupCudaArray();
    upload();
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid image element type on compressedImage2D sampler (%s); "
        "must be ANARI_UINT8 or ANARI_INT8",
        anari::toString(elementType));
    return;
  }

  const auto format = parseFormat(m_format);
  if (!format) {
    cleanupCudaArray();
    upload();
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid format '%s' on compressedImage2D sampler",
        m_format.c_str());
    return;
  }

  if (m_size.x == 0 || m_size.y == 0) {
    cleanupCudaArray();
    upload();
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing or invalid 'size' on compressedImage2D sampler");
    return;
  }

  const auto modified = m_image->lastDataModified();
  if (!m_cuArray || m_cuArrayUploaded < modified || m_cuArrayFormat != m_format
      || m_cuArraySize != m_size) {
    cleanupCudaArray();
    const auto error = makeCudaCompressedTextureArray(m_cuArray,
        m_size,
        m_image->begin(AddressSpace::HOST) ? m_image->begin(AddressSpace::HOST)
                                           : m_image->begin(AddressSpace::GPU),
        m_image->size() * anari::sizeOf(elementType),
        format->kind);
    if (error != cudaSuccess) {
      // Handled here; do not leave it for a later CUDA call to pick up.
      cudaGetLastError();
      if (m_cuArrayFailure != modified) {
        m_cuArrayFailure = modified;
        reportMessage(ANARI_SEVERITY_ERROR,
            "compressedImage2D upload failed (%s %ux%u, %zu bytes): %s",
            m_format.c_str(),
            m_size.x,
            m_size.y,
            m_image->size() * anari::sizeOf(elementType),
            cudaGetErrorString(error));
      }
      upload();
      return;
    }
    m_cuArrayUploaded = modified;
    m_cuArrayFormat = m_format;
    m_cuArraySize = m_size;
  }

  m_texture = makeCudaCompressedTextureObject2D(
      m_cuArray, format->kind, m_filter, m_wrap1, m_wrap2, m_borderColor);
  m_texels = makeCudaCompressedTexelObject2D(
      m_cuArray, format->kind, "nearest", m_wrap1, m_wrap2, m_borderColor);
  if (!m_texture || !m_texels)
    cleanupTextureObjects();

  upload();
}

bool CompressedImage2D::isValid() const
{
  return m_texture && m_texels;
}

int CompressedImage2D::numChannels() const
{
  const auto format = parseFormat(m_format);
  return format ? format->numChannels : 0;
}

SamplerGPUData CompressedImage2D::gpuData() const
{
  SamplerGPUData retval = Sampler::gpuData();
  if (!isValid())
    return retval;
  retval.type = SamplerType::TEXTURE2D;
  retval.image2D.texobj = m_texture;
  retval.image2D.texelTexobj = m_texels;
  retval.image2D.size = m_cuArraySize;
  retval.image2D.invSize =
      glm::vec2(1.0f / m_cuArraySize.x, 1.0f / m_cuArraySize.y);

  return retval;
}

void CompressedImage2D::cleanupTextureObjects()
{
  if (m_texels)
    cudaDestroyTextureObject(m_texels);
  if (m_texture)
    cudaDestroyTextureObject(m_texture);
  m_texels = {};
  m_texture = {};
}

void CompressedImage2D::cleanupCudaArray()
{
  // Texture objects must be gone before their storage.
  cleanupTextureObjects();
  if (m_cuArray)
    cudaFreeArray(m_cuArray);
  m_cuArray = {};
  m_cuArrayUploaded = {};
  m_cuArrayFormat.clear();
  m_cuArraySize = {};
}

} // namespace visrtx
