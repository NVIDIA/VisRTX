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

#include "SamplerRegistry.h"

#include "array/Array2D.h"
#include "array/Array3D.h"
#include "libmdl/ArgumentBlockDescriptor.h"
#include "optix_visrtx.h"
#include "scene/surface/material/sampler/CompressedImage2D.h"
#include "scene/surface/material/sampler/Image2D.h"
#include "scene/surface/material/sampler/Image3D.h"

#include <anari/frontend/anari_enums.h>
#include <helium/utility/IntrusivePtr.h>
#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/itexture.h>
#include <mi/neuraylib/itile.h>
#include <mi/neuraylib/itransaction.h>

#include <fmt/format.h>

#include <fstream>
#include <glm/ext/vector_uint2_sized.hpp>
#include <string>
#include <string_view>

#include <stb_image.h>
#include "dds.h"

using namespace std::string_view_literals;

using U64Vec2 = glm::u64vec2;
namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(U64Vec2, ANARI_UINT64_VEC2);
}

namespace visrtx::mdl {

SamplerRegistry::SamplerRegistry(
    libmdl::Core *core, DeviceGlobalState *deviceState)
    : m_core(core), m_deviceState(deviceState)
{}

SamplerRegistry::~SamplerRegistry()
{
  if (!m_dbToSampler.empty()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "SamplerRegistry is not empty on destruction");
  }
}

Sampler *SamplerRegistry::loadFromDDS(
    const std::string_view &filePath, libmdl::ColorSpace colorSpace)
{
  std::ifstream ifs(std::string(filePath), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
        "Failed to open file '{}'",
        filePath);
    return {};
  }

  std::vector<char> buffer(
      (std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  auto dds = reinterpret_cast<const dds::DdsFile *>(data(buffer));
  if (dds->magic != dds::DDS_MAGIC
      || dds->header.size != sizeof(dds::DdsHeader)) {
    m_core->logMessage(
        mi::base::MESSAGE_SEVERITY_WARNING, "Invalid DDS file '{}'", filePath);
    return {};
  }

  // Check if we have a dxt10 header
  constexpr const auto baseReqFlags = dds::DDSD_CAPS | dds::DDSD_HEIGHT
      | dds::DDSD_WIDTH | dds::DDSD_PIXELFORMAT;
  if ((dds->header.flags & baseReqFlags) != baseReqFlags) {
    m_core->logMessage(
        mi::base::MESSAGE_SEVERITY_WARNING, "Invalid DDS file '{}'", filePath);
    return {};
  }

  constexpr const auto textureReqFlags = dds::DDSCAPS_TEXTURE;
  if ((dds->header.caps & textureReqFlags) != textureReqFlags) {
    m_core->logMessage(
        mi::base::MESSAGE_SEVERITY_WARNING, "Invalid DDS file '{}'", filePath);
    return {};
  }

  const char *compressedFormat = {};
  const char *format = {};
  bool alpha = dds->header.pixelFormat.flags & dds::DDPF_ALPHAPIXELS;
  auto dxgiFormat = dds::getDxgiFormat(dds);
  switch (dxgiFormat) {
  case dds::DXGI_FORMAT_BC1_UNORM: {
    // BC1: RGB/RGBA, 1bit alpha
    if (colorSpace == libmdl::ColorSpace::sRGB) {
      compressedFormat = alpha ? "BC1_RGBA_SRGB" : "BC1_RGB_SRGB";
    } else {
      compressedFormat = alpha ? "BC1_RGBA" : "BC1_RGB";
    }
    break;
  }
  case dds::DXGI_FORMAT_BC1_UNORM_SRGB: {
    // BC1: RGB/RGBA, 1bit alpha
    if (colorSpace == libmdl::ColorSpace::sRGB) {
      compressedFormat = alpha ? "BC1_RGBA_SRGB" : "BC1_RGB_SRGB";
    } else {
      compressedFormat = alpha ? "BC1_RGBA" : "BC1_RGB";
    }
    break;
  }
  case dds::DXGI_FORMAT_BC2_UNORM: {
    // BC2: RGB/RGBA, 4bit alpha
    if (colorSpace == libmdl::ColorSpace::sRGB) {
      compressedFormat = "BC2_SRGB";
    } else {
      compressedFormat = "BC2";
    }
    break;
  }
  case dds::DXGI_FORMAT_BC2_UNORM_SRGB: {
    // BC2: RGB/RGBA, 4bit alpha
    compressedFormat = "BC2_SRGB";
    break;
  }
  case dds::DXGI_FORMAT_BC3_UNORM: {
    // BC3: RGB/RGBA, 8bit alpha
    if (colorSpace == libmdl::ColorSpace::sRGB) {
      compressedFormat = "BC3_SRGB";
    } else {
      compressedFormat = "BC3";
    }
    break;
  }
  case dds::DXGI_FORMAT_BC3_UNORM_SRGB: {
    // BC3: RGB/RGBA, 8bit alpha
    compressedFormat = "BC3_SRGB";
    break;
  }
  case dds::DXGI_FORMAT_BC4_UNORM: {
    // BC4: R/RG
    compressedFormat = "BC4";
    break;
  }
  case dds::DXGI_FORMAT_BC4_SNORM: {
    // BC4: R/RG
    compressedFormat = "BC4_SNORM";
    break;
  }
  case dds::DXGI_FORMAT_BC5_UNORM: {
    // BC5: RG/RGBA
    compressedFormat = "BC5";
    break;
  }
  case dds::DXGI_FORMAT_BC5_SNORM: {
    // BC5: RG/RGBA
    compressedFormat = "BC5_SNORM";
    break;
  }
  case dds::DXGI_FORMAT_BC6H_UF16: {
    // BC6H: RGB
    compressedFormat = "BC6H_UFLOAT";
    break;
  }
  case dds::DXGI_FORMAT_BC6H_SF16: {
    // BC6H: RGB
    compressedFormat = "BC6H_SFLOAT";
    break;
  }
  case dds::DXGI_FORMAT_BC7_UNORM: {
    // BC7: RGB/RGBA
    if (colorSpace == libmdl::ColorSpace::sRGB) {
      compressedFormat = "BC7_SRGB";
    } else {
      compressedFormat = "BC7";
    }
    break;
  }
  case dds::DXGI_FORMAT_BC7_UNORM_SRGB: {
    // BC7: RGB/RGBA
    compressedFormat = "BC7_SRGB";
    break;
  }
  case dds::DXGI_FORMAT_R8G8B8A8_UNORM: {
    // RGBA8
    format = "RGBA8";
    break;
  }
  case dds::DXGI_FORMAT_B8G8R8A8_UNORM: {
    // RGBA8
    format = "BGRA8";
    break;
  }
  case dds::DXGI_FORMAT_UNKNOWN: {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
        "Cannot guess DDS format for file '{}'",
        filePath);
    break;
  }
  default: {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
        "unsupported DDS format '{}' for file '{}'",
        dds::getDxgiFormatString(dxgiFormat),
        filePath);
    break;
  }
  }

  Sampler *tex = {};

  if (compressedFormat) {
    // Simple  implementation that only handling single level mipmaps
    // and non cubemap textures.
    auto linearSize = dds::computeLinearSize(dds);

    if ((dds->header.flags & dds::DDSD_LINEARSIZE)
        && (linearSize != dds->header.pitchOrLinearSize)) {
      m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
          "Ignoring invalid linear size {} (should be {}) for compressed texture '{}'",
          dds->header.pitchOrLinearSize,
          linearSize,
          filePath);
    }

    Array1DMemoryDescriptor desc = {
        {
            dds::getDataPointer(dds),
            nullptr,
            nullptr,
            ANARI_UINT8,
        },
        static_cast<uint64_t>(linearSize),
    };

    auto array1d = new Array1D(m_deviceState, desc);
    array1d->commitParameters();
    array1d->uploadArrayData();
    array1d->finalize();
    auto image2d = new CompressedImage2D(m_deviceState);
    image2d->setParam("image", array1d);
    image2d->setParam("format", std::string(compressedFormat));
    image2d->setParam("size", U64Vec2(dds->header.width, dds->header.height));
    array1d->refDec(helium::PUBLIC);

    image2d->commitParameters();
    image2d->finalize();
    tex = image2d;
  } else if (format) {
    anari::DataType texelType = ANARI_UNKNOWN;

    Array2DMemoryDescriptor desc = {
        {
            dds::getDataPointer(dds),
            nullptr,
            nullptr,
            ANARI_UFIXED8_VEC4,
        },
        dds->header.width,
        dds->header.height,
    };

    auto array2d = new Array2D(m_deviceState, desc);
    array2d->commitParameters();
    array2d->finalize();
    array2d->uploadArrayData();
    auto image2d = new Image2D(m_deviceState);
    image2d->setParam("image", array2d);
    array2d->refDec(helium::PUBLIC);

    image2d->commitParameters();
    image2d->finalize();
    tex = image2d;
  } else {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
        "Unsupported texture format for '{}'",
        filePath);
  }

  return tex;
}

Sampler *SamplerRegistry::loadFromImage(
    const std::string_view &filePath, libmdl::ColorSpace colorSpace)
{
  auto filePathS = std::string(filePath);
  auto isHdr = stbi_is_hdr(filePathS.c_str());

  int width, height, n;
  void *data = nullptr;

  data = isHdr ? static_cast<void *>(
                     stbi_loadf(filePathS.c_str(), &width, &height, &n, 0))
               : static_cast<void *>(
                     stbi_load(filePathS.c_str(), &width, &height, &n, 0));

  if (!data || n < 1) {
    m_core->logMessage(mi::base::details::MESSAGE_SEVERITY_WARNING,
        "Failed to load texture '{}'",
        filePath);
    return {};
  }

  int texelType = isHdr
      ? ANARI_FLOAT32_VEC4
      : (colorSpace == libmdl::ColorSpace::Linear ? ANARI_UFIXED8_VEC4
                                                  : ANARI_UFIXED8_RGBA_SRGB);
  if (n == 3)
    texelType = isHdr
        ? ANARI_FLOAT32_VEC3
        : (colorSpace == libmdl::ColorSpace::Linear ? ANARI_UFIXED8_VEC3
                                                    : ANARI_UFIXED8_RGB_SRGB);
  else if (n == 2)
    texelType = isHdr
        ? ANARI_FLOAT32_VEC2
        : (colorSpace == libmdl::ColorSpace::Linear ? ANARI_UFIXED8_VEC2
                                                    : ANARI_UFIXED8_RA_SRGB);
  else if (n == 1)
    texelType = isHdr
        ? ANARI_FLOAT32
        : (colorSpace == libmdl::ColorSpace::Linear ? ANARI_UFIXED8
                                                    : ANARI_UFIXED8_R_SRGB);

  Array2DMemoryDescriptor desc = {
      {
          data,
          nullptr,
          nullptr,
          texelType,
      },
      static_cast<uint64_t>(width),
      static_cast<uint64_t>(height),
  };

  auto array2d = new Array2D(m_deviceState, desc);
  array2d->commitParameters();
  array2d->uploadArrayData();
  array2d->finalize();
  auto image2d = new Image2D(m_deviceState);
  image2d->setParam("image", array2d);
  image2d->commitParameters();
  image2d->finalize();
  array2d->refDec(helium::PUBLIC);

  return image2d;
}

Sampler *SamplerRegistry::loadFromFile(
    const std::string_view &filePath, libmdl::ColorSpace colorSpace)
{
  if (size(filePath) > 4 && filePath.substr(size(filePath) - 4) == ".dds") {
    return loadFromDDS(filePath, colorSpace);
  } else {
    return loadFromImage(filePath, colorSpace);
  }
}

Sampler *SamplerRegistry::loadFromTextureDesc(
    const libmdl::TextureDescriptor &textureDesc)
{
  switch (textureDesc.shape) {
  case libmdl::Shape::TwoD: {
    return loadFromImage(textureDesc.url, textureDesc.colorSpace);
  }
  case libmdl::Shape::BsdfData: {
    auto texelType = ANARI_FLOAT32_VEC4;

    if (textureDesc.bsdf.pixelFormat == "Sint8"sv) {
      texelType = ANARI_UFIXED8;
    } else if (textureDesc.bsdf.pixelFormat == "Sint32"sv) {
      texelType = ANARI_UFIXED32;
    } else if (textureDesc.bsdf.pixelFormat == "Float32"sv) {
      texelType = ANARI_FLOAT32;
    } else if (textureDesc.bsdf.pixelFormat == "Float32<2>"sv) {
      texelType = ANARI_FLOAT32_VEC2;
    } else if (textureDesc.bsdf.pixelFormat == "Float32<3>"sv) {
      texelType = ANARI_FLOAT32_VEC3;
    } else if (textureDesc.bsdf.pixelFormat == "Float32<4>"sv) {
      texelType = ANARI_FLOAT32_VEC4;
    } else if (textureDesc.bsdf.pixelFormat == "Rgb"sv) {
      texelType = ANARI_UFIXED8_VEC3;
    } else if (textureDesc.bsdf.pixelFormat == "Rgba"sv) {
      texelType = ANARI_UFIXED8_VEC4;
    } else if (textureDesc.bsdf.pixelFormat == "Rgbe"sv) {
      texelType = ANARI_UFIXED8_VEC4;
    } else if (textureDesc.bsdf.pixelFormat == "Rgbea"sv) {
      texelType = ANARI_UNKNOWN;
    } else if (textureDesc.bsdf.pixelFormat == "Rgb_16"sv) {
      texelType = ANARI_UFIXED16_VEC3;
    } else if (textureDesc.bsdf.pixelFormat == "Rgba_16"sv) {
      texelType = ANARI_UFIXED16_VEC4;
    } else if (textureDesc.bsdf.pixelFormat == "Rgb_fp"sv) {
      texelType = ANARI_FLOAT32_VEC3;
    } else if (textureDesc.bsdf.pixelFormat == "Color"sv) {
      texelType = ANARI_UFIXED8_VEC3;
    }

    Array3DMemoryDescriptor desc = {
        {
            textureDesc.bsdf.data,
            nullptr,
            nullptr,
            texelType,
        },
        textureDesc.bsdf.dims[0],
        textureDesc.bsdf.dims[1],
        textureDesc.bsdf.dims[2],
    };

    auto array3d = new Array3D(m_deviceState, desc);
    array3d->commitParameters();
    array3d->uploadArrayData();
    auto image3d = new Image3D(m_deviceState);
    image3d->setParam("image", array3d);
    image3d->commitParameters();
    image3d->finalize();
    array3d->refDec(helium::PUBLIC);

    return image3d;
  }
  }
  return {};
}

Sampler *SamplerRegistry::acquireSampler(
    const std::string &filePath, libmdl::ColorSpace colorSpace)
{
  if (auto it = m_dbToSampler.find(filePath); it != end(m_dbToSampler)) {
    it->second->refInc();
    return it->second;
  }

  auto sampler = loadFromFile(filePath, colorSpace);
  if (sampler) {
    sampler->refInc();
    sampler->refDec(helium::PUBLIC); // Drop the implicit public refcount that
                                     // we don't rely on.
    m_dbToSampler.insert({filePath, sampler});
  } else {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Unable to create sampler for texture `{}`",
        filePath);
  }

  return sampler;
}

Sampler *SamplerRegistry::acquireSampler(
    const libmdl::TextureDescriptor &textureDesc)
{
  if (auto it = m_dbToSampler.find(textureDesc.url); it != end(m_dbToSampler)) {
    it->second->refInc();
    return it->second;
  }

  auto sampler = loadFromTextureDesc(textureDesc);
  if (sampler) {
    sampler->refInc();
    sampler->refDec(helium::PUBLIC); // Drop the implicit public refcount that
                                     // we don't rely on.
    m_dbToSampler.insert({textureDesc.url, sampler});
  } else {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Unable to create sampler for texture db name `{}`",
        textureDesc.url);
  }

  return sampler;
}

bool SamplerRegistry::releaseSampler(const Sampler *sampler)
{
  if (auto it = std::find_if(std::begin(m_dbToSampler),
          std::end(m_dbToSampler),
          [sampler](const auto &p) { return p.second == sampler; });
      it != std::end(m_dbToSampler)) {
    auto useCount = it->second->useCount(helium::INTERNAL);
    it->second->refDec();
    if (useCount == 1) {
      // Our counter dropped to 1 pre-refDec, no one else is using the
      // sampler.
      m_dbToSampler.erase(it);
      return true;
    }
  } else {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Removing an unknown sampler {}\n",
        fmt::ptr(sampler));
  }

  return false;
}

} // namespace visrtx::mdl