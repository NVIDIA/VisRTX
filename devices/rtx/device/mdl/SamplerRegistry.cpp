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

#include "SamplerRegistry.h"

#include "MdlCompileCoordinator.h"
#include "array/Array2D.h"
#include "array/Array3D.h"
#include "libmdl/ArgumentBlockDescriptor.h"
#include "optix_visrtx.h"
#include "sampler/CompressedImage2D.h"
#include "sampler/Image2D.h"
#include "sampler/Image3D.h"

#include <future>
#include <unordered_set>

#include <anari/frontend/anari_enums.h>
#include <cassert>
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

    // Move the file bytes to the heap and hand them to the array (CAPTURED)
    // instead of borrowing the soon-freed local `buffer` (SHARED), which would
    // force helium to privatize a host copy + log "making private copy of
    // shared host array". appMemory is an offset into the buffer, so the array
    // can't own it directly; pass the owning vector as deleterPtr and free that.
    auto *owned = new std::vector<char>(std::move(buffer));
    Array1DMemoryDescriptor desc = {
        {
            dds::getDataPointer(dds),
            [](const void *userPtr, const void * /*appMemory*/) {
              delete static_cast<const std::vector<char> *>(userPtr);
            },
            owned,
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

    // Registry-internal objects never pass through anariCommitParameters(),
    // so nothing ever captures their committed snapshot — a later buffered
    // re-commit (change-observer notification) would run under a
    // ReadCommittedScope against an EMPTY snapshot and lose every parameter.
    // Mirror staging into the snapshot explicitly.
    image2d->snapshotParameters();
    image2d->commitParameters();
    image2d->finalize();
    tex = image2d;
  } else if (format) {
    anari::DataType texelType = ANARI_UNKNOWN;

    // See the compressed branch: own the file bytes (CAPTURED) rather than
    // borrowing the local buffer, to avoid the per-texture privatize copy.
    auto *owned = new std::vector<char>(std::move(buffer));
    Array2DMemoryDescriptor desc = {
        {
            dds::getDataPointer(dds),
            [](const void *userPtr, const void * /*appMemory*/) {
              delete static_cast<const std::vector<char> *>(userPtr);
            },
            owned,
            ANARI_UFIXED8_VEC4,
        },
        dds->header.width,
        dds->header.height,
    };

    auto array2d = new Array2D(m_deviceState, desc);
    array2d->commitParameters();
    array2d->finalize();
    // No uploadArrayData(): the linear device buffer is never sampled. Image2D
    // builds its cudaArray from the host data, so uploading a redundant linear
    // copy just doubles this texture's VRAM footprint.
    auto image2d = new Image2D(m_deviceState);
    image2d->setParam("image", array2d);
    array2d->refDec(helium::PUBLIC);

    image2d->snapshotParameters(); // see the compressed-path comment above
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

void SamplerRegistry::StbDeleter::operator()(void *p) const
{
  stbi_image_free(p);
}

SamplerRegistry::StagedImage SamplerRegistry::decodeStb(
    const std::string_view &filePath)
{
  auto filePathS = std::string(filePath);
  auto isHdr = stbi_is_hdr(filePathS.c_str());

  int width, height, n;
  void *data = isHdr ? static_cast<void *>(
                     stbi_loadf(filePathS.c_str(), &width, &height, &n, 0))
                     : static_cast<void *>(
                         stbi_load(filePathS.c_str(), &width, &height, &n, 0));

  StagedImage image;
  if (!data || n < 1) {
    if (data)
      stbi_image_free(data);
    return image;
  }
  image.data.reset(data);
  image.width = width;
  image.height = height;
  image.channels = n;
  image.isHdr = isHdr;
  return image;
}

Sampler *SamplerRegistry::loadFromImage(
    const std::string_view &filePath, libmdl::ColorSpace colorSpace)
{
  auto image = decodeStb(filePath);
  if (!image.data) {
    m_core->logMessage(mi::base::details::MESSAGE_SEVERITY_WARNING,
        "Failed to load texture '{}'",
        filePath);
    return {};
  }
  return createFromStb(std::move(image), colorSpace);
}

Sampler *SamplerRegistry::createFromStb(
    StagedImage image, libmdl::ColorSpace colorSpace)
{
  const int width = image.width;
  const int height = image.height;
  const int n = image.channels;
  const bool isHdr = image.isHdr;

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

  // Hand the stb_image allocation to the Array2D (CAPTURED ownership) instead
  // of borrowing it (SHARED). Borrowing forces helium to privatize a host copy
  // when the public ref is dropped while the sampler still references it -- a
  // per-texture deep copy + "making private copy of shared host array" warning,
  // and it also leaked the buffer (nothing freed the stb allocation). With a
  // deleter the array owns and frees it: no copy, no warning, no leak. Release
  // it from the StagedImage so ownership transfers cleanly (no double free).
  Array2DMemoryDescriptor desc = {
      {
          image.data.release(),
          [](const void * /*userPtr*/, const void *appMemory) {
            stbi_image_free(const_cast<void *>(appMemory));
          },
          nullptr,
          texelType,
      },
      static_cast<uint64_t>(width),
      static_cast<uint64_t>(height),
  };

  auto array2d = new Array2D(m_deviceState, desc);
  array2d->commitParameters();
  array2d->finalize();
  // No uploadArrayData(): the linear device buffer is never sampled. Image2D
  // builds its cudaArray from the host data, so uploading a redundant linear
  // copy just doubles this texture's VRAM footprint.
  auto image2d = new Image2D(m_deviceState);
  image2d->setParam("image", array2d);
  image2d->snapshotParameters(); // see the compressed-path comment above
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

    // df_data is MDL-SDK-owned, process-stable built-in table data shared
    // across materials (the registry caches the sampler by this data pointer).
    // Borrowing it with a null deleter (SHARED) made helium privatize a
    // redundant host copy + log "making private copy of shared host array". A
    // no-op deleter marks it CAPTURED so no copy is made; we must NOT free
    // SDK-owned memory, hence the empty deleter rather than free/delete.
    Array3DMemoryDescriptor desc = {
        {
            textureDesc.bsdf.data,
            [](const void *, const void *) {},
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
    image3d->snapshotParameters(); // see the compressed-path comment above
    image3d->commitParameters();
    image3d->finalize();
    array3d->refDec(helium::PUBLIC);

    return image3d;
  }
  }
  return {};
}

// The same image file can be requested in different color spaces (e.g. a map
// used as both sRGB base color and raw data). The decoded sampler differs, so
// the color space must be part of the cache key.
static std::string samplerCacheKey(
    std::string_view url, libmdl::ColorSpace colorSpace)
{
  return (colorSpace == libmdl::ColorSpace::sRGB ? "srgb:" : "linear:")
      + std::string(url);
}

void SamplerRegistry::decodeToStaging(
    const libmdl::TextureDescriptor &textureDesc)
{
  // Only plain image files are staged: BSDF-data tables come from the SDK and
  // .dds is handed to the device as-is -- both take the inline path.
  if (textureDesc.shape == libmdl::Shape::BsdfData)
    return;
  const auto &url = textureDesc.url;
  if (url.empty() || (url.size() > 4 && url.substr(url.size() - 4) == ".dds"))
    return;

  const auto key = samplerCacheKey(url, textureDesc.colorSpace);
  {
    std::lock_guard<std::mutex> guard(m_stagingMutex);
    if (m_staging.count(key))
      return; // another texture slot in this flush already staged it
  }
  auto image = decodeStb(url);
  if (!image.data)
    return; // decode failed -> acquireSampler falls back to the inline path
  std::lock_guard<std::mutex> guard(m_stagingMutex);
  m_staging.try_emplace(key, std::move(image));
}

SamplerRegistry::StagedImage SamplerRegistry::takeStaged(const std::string &key)
{
  std::lock_guard<std::mutex> guard(m_stagingMutex);
  auto it = m_staging.find(key);
  if (it == end(m_staging))
    return {};
  StagedImage image = std::move(it->second);
  m_staging.erase(it);
  return image;
}

void SamplerRegistry::stageDecodeBatch(MdlCompileCoordinator &coordinator,
    const std::vector<libmdl::TextureDescriptor> &descriptors)
{
  // Runs on the commit thread, which is the sole mutator of m_dbToSampler and is
  // then blocked in the waits below, so reading the sampler cache here needs no
  // lock. Skip textures already cached and collapse repeated keys, so each
  // unique uncached texture is decoded exactly once -- a texture referenced N
  // times (in one material or shared across the flush) is not decoded N times.
  std::unordered_set<std::string> scheduled;
  std::vector<std::future<void>> decodes;
  for (const auto &desc : descriptors) {
    if (desc.shape == libmdl::Shape::BsdfData || desc.url.empty())
      continue;
    if (desc.url.size() > 4 && desc.url.substr(desc.url.size() - 4) == ".dds")
      continue;
    auto key = samplerCacheKey(desc.url, desc.colorSpace);
    {
      std::lock_guard<std::mutex> guard(m_cacheMutex);
      if (m_dbToSampler.count(key))
        continue; // already have the sampler; acquireSampler will cache-hit
    }
    if (!scheduled.insert(std::move(key)).second)
      continue; // this key is already being decoded in this batch
    decodes.push_back(coordinator.submit(
        [this, desc] { decodeToStaging(desc); }));
  }
  for (auto &decode : decodes)
    decode.get();
}

Sampler *SamplerRegistry::acquireSampler(
    const std::string &filePath, libmdl::ColorSpace colorSpace)
{
  auto key = samplerCacheKey(filePath, colorSpace);
  {
    std::lock_guard<std::mutex> guard(m_cacheMutex);
    if (auto it = m_dbToSampler.find(key); it != end(m_dbToSampler)) {
      it->second.acquires++;
      it->second.sampler->refInc();
      return it->second.sampler;
    }
  }

  // Load outside the lock (device work); acquires are flush-serialized, so no
  // other acquire inserts this key meanwhile and release only touches existing
  // entries -- the insert below is safe.
  auto sampler = loadFromFile(filePath, colorSpace);
  if (sampler) {
    sampler->refInc();
    sampler->refDec(helium::PUBLIC); // Drop the implicit public refcount that
                                     // we don't rely on.
    std::lock_guard<std::mutex> guard(m_cacheMutex);
    m_dbToSampler.insert({key, {sampler, 1}});
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
  auto key = samplerCacheKey(textureDesc.url, textureDesc.colorSpace);
  // Take (and clear) any staged pixels for this key up front, so a staged
  // texture that turns out to be a cache hit is not left lingering.
  StagedImage staged = takeStaged(key);
  {
    std::lock_guard<std::mutex> guard(m_cacheMutex);
    if (auto it = m_dbToSampler.find(key); it != end(m_dbToSampler)) {
      it->second.acquires++;
      it->second.sampler->refInc();
      return it->second.sampler;
    }
  }

  auto sampler = staged.data
      ? createFromStb(std::move(staged), textureDesc.colorSpace)
      : loadFromTextureDesc(textureDesc);
  if (sampler) {
    sampler->refInc();
    sampler->refDec(helium::PUBLIC); // Drop the implicit public refcount that
                                     // we don't rely on.
    std::lock_guard<std::mutex> guard(m_cacheMutex);
    m_dbToSampler.insert({key, {sampler, 1}});
  } else {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Unable to create sampler for texture db name `{}`",
        textureDesc.url);
  }

  return sampler;
}

bool SamplerRegistry::releaseSampler(const Sampler *sampler)
{
  // Runs on whatever thread drops the material's last reference (any app thread
  // under khr_device_synchronization), so guard the cache against a concurrent
  // acquire on the commit thread.
  std::lock_guard<std::mutex> guard(m_cacheMutex);
  if (auto it = std::find_if(std::begin(m_dbToSampler),
          std::end(m_dbToSampler),
          [sampler](const auto &p) { return p.second.sampler == sampler; });
      it != std::end(m_dbToSampler)) {
    // A double-release while other acquires are outstanding would silently
    // corrupt the count (early erase + a leaked reference for the wronged
    // holder) — fail fast in debug builds.
    assert(it->second.acquires > 0);
    it->second.acquires--;
    it->second.sampler->refDec();
    if (it->second.acquires == 0) {
      // Last REGISTRY acquire gone: drop the cache entry now, whether or not
      // another holder (deferred commit buffer, ...) keeps the object alive a
      // little longer. Deciding from the object's refcount instead left this
      // entry dangling once that holder dropped the true last reference —
      // use-after-free on the next same-key acquire.
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