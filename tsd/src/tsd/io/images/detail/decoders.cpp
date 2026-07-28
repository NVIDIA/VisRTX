// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/images/detail/decoders.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/importers/detail/dds.h"
#include "tsd/io/importers/detail/importer_common.hpp"
// stb_image
#include "stb_image.h"
#ifndef _WIN32
#include "tinyexr.h"
#endif
#if TSD_USE_OIIO
// OpenImageIO
#include <OpenImageIO/imageio.h>
#endif
// std
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

namespace tsd::io::detail {

using namespace tsd::core;

namespace {

std::string lowered(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  return s;
}

// Decoded texels are always float, so channel count alone picks the type.
anari::DataType texelTypeForChannelCount(int numChannels)
{
  switch (numChannels) {
  case 1:
    return ANARI_FLOAT32;
  case 2:
    return ANARI_FLOAT32_VEC2;
  case 3:
    return ANARI_FLOAT32_VEC3;
  default:
    return ANARI_FLOAT32_VEC4;
  }
}

DecodedImage decodeStb(
    const void *data, size_t numBytes, ColorSpace colorSpace, const char *id)
{
  int width = 0;
  int height = 0;
  int n = 0;

  stbi_ldr_to_hdr_scale(1.0f);
  stbi_ldr_to_hdr_gamma(colorSpace == ColorSpace::LINEAR ? 1.0f : 2.2f);

  float *decoded = stbi_loadf_from_memory(static_cast<const stbi_uc *>(data),
      int(numBytes),
      &width,
      &height,
      &n,
      0);

  if (!decoded) {
    logError("[decodeImage] failed to decode image '%s'", id);
    return {};
  }
  if (n < 1) {
    logWarning("[decodeImage] image '%s' with %i channels not imported", id, n);
    stbi_image_free(decoded);
    return {};
  }

  DecodedImage image;
  image.elementType = texelTypeForChannelCount(n);
  image.width = size_t(width);
  image.height = size_t(height);
  // stb hands back the picture's first row first, whatever the container's own
  // storage order was -- it undoes BMP's and TGA's bottom-up layouts itself.
  image.rowOrder = RowOrder::TOP_DOWN;
  const size_t numBytesOut =
      size_t(width) * size_t(height) * size_t(n) * sizeof(float);
  image.texels.assign(reinterpret_cast<const char *>(decoded),
      reinterpret_cast<const char *>(decoded) + numBytesOut);

  stbi_image_free(decoded);
  return image;
}

DecodedImage decodeDds(const void *data, size_t numBytes, const char *id)
{
  if (numBytes < sizeof(dds::DdsFile)) {
    logError("[decodeImage] invalid DDS buffer '%s'", id);
    return {};
  }

  auto *file = reinterpret_cast<const dds::DdsFile *>(data);
  if (file->magic != dds::DDS_MAGIC
      || file->header.size != sizeof(dds::DdsHeader)) {
    logError("[decodeImage] invalid DDS buffer '%s'", id);
    return {};
  }

  constexpr auto baseReqFlags = dds::DDSD_CAPS | dds::DDSD_HEIGHT
      | dds::DDSD_WIDTH | dds::DDSD_PIXELFORMAT;
  if ((file->header.flags & baseReqFlags) != baseReqFlags) {
    logError("[decodeImage] invalid DDS buffer '%s'", id);
    return {};
  }

  if ((file->header.caps & dds::DDSCAPS_TEXTURE) != dds::DDSCAPS_TEXTURE) {
    logError("[decodeImage] invalid DDS buffer '%s'", id);
    return {};
  }

  const bool alpha = file->header.pixelFormat.flags & dds::DDPF_ALPHAPIXELS;
  Token compressedFormat = {};
  switch (dds::getDxgiFormat(file)) {
  case dds::DXGI_FORMAT_BC1_UNORM:
    // BC1: RGB/RGBA, 1bit alpha
    compressedFormat = alpha ? "BC1_RGBA" : "BC1_RGB";
    break;
  case dds::DXGI_FORMAT_BC1_UNORM_SRGB:
    compressedFormat = alpha ? "BC1_RGBA_SRGB" : "BC1_RGB_SRGB";
    break;
  case dds::DXGI_FORMAT_BC2_UNORM:
    compressedFormat = "BC2";
    break;
  case dds::DXGI_FORMAT_BC2_UNORM_SRGB:
    compressedFormat = "BC2_SRGB";
    break;
  case dds::DXGI_FORMAT_BC3_UNORM:
    compressedFormat = "BC3";
    break;
  case dds::DXGI_FORMAT_BC3_UNORM_SRGB:
    compressedFormat = "BC3_SRGB";
    break;
  case dds::DXGI_FORMAT_BC4_UNORM:
    compressedFormat = "BC4";
    break;
  case dds::DXGI_FORMAT_BC4_SNORM:
    compressedFormat = "BC4_SNORM";
    break;
  case dds::DXGI_FORMAT_BC5_UNORM:
    compressedFormat = "BC5";
    break;
  case dds::DXGI_FORMAT_BC5_SNORM:
    compressedFormat = "BC5_SNORM";
    break;
  case dds::DXGI_FORMAT_BC6H_UF16:
    compressedFormat = "BC6H_UFLOAT";
    break;
  case dds::DXGI_FORMAT_BC6H_SF16:
    compressedFormat = "BC6H_SFLOAT";
    break;
  case dds::DXGI_FORMAT_BC7_UNORM:
    compressedFormat = "BC7";
    break;
  case dds::DXGI_FORMAT_BC7_UNORM_SRGB:
    compressedFormat = "BC7_SRGB";
    break;
  default:
    logError("[decodeImage] unsupported DDS format '%c%c%c%c' for '%s'",
        file->header.pixelFormat.fourCC & 0xff,
        (file->header.pixelFormat.fourCC >> 8) & 0xff,
        (file->header.pixelFormat.fourCC >> 16) & 0xff,
        (file->header.pixelFormat.fourCC >> 24) & 0xff,
        id);
    return {};
  }

  // Simple implementation that only handles single level mipmaps and
  // non-cubemap textures.
  const auto linearSize = dds::computeLinearSize(file);
  if ((file->header.flags & dds::DDSD_LINEARSIZE)
      && (linearSize != file->header.pitchOrLinearSize)) {
    logError(
        "[decodeImage] ignoring invalid linear size %u (should be %u) for compressed texture '%s'",
        file->header.pitchOrLinearSize,
        linearSize,
        id);
  }

  DecodedImage image;
  image.elementType = ANARI_INT8;
  image.width = file->header.width;
  image.height = file->header.height;
  image.blockCompressed = true;
  image.compressedFormat = compressedFormat;
  // Block-compressed rows come in 4x4 groups, so this is the order the file
  // authored and the only order the blocks can be handed on in.
  image.rowOrder = RowOrder::TOP_DOWN;
  const auto *blocks =
      reinterpret_cast<const char *>(dds::getDataPointer(file));
  image.texels.assign(blocks, blocks + linearSize);

  return image;
}

#ifndef _WIN32
// Follow actual HDRI importer: tinyexr is excluded on Windows; to be
// investigated.
DecodedImage decodeExr(const std::string &path)
{
  float *rgba = nullptr;
  int width = 0;
  int height = 0;
  const char *err = nullptr;

  if (LoadEXR(&rgba, &width, &height, path.c_str(), &err) != TINYEXR_SUCCESS) {
    logError("[decodeImage] failed to load EXR '%s': %s",
        path.c_str(),
        err ? err : "unknown error");
    if (err)
      FreeEXRErrorMessage(err);
    return {};
  }

  DecodedImage image;
  image.elementType = ANARI_FLOAT32_VEC4;
  image.width = size_t(width);
  image.height = size_t(height);
  image.rowOrder = RowOrder::TOP_DOWN;
  const auto *bytes = reinterpret_cast<const char *>(rgba);
  image.texels.assign(
      bytes, bytes + size_t(width) * size_t(height) * 4 * sizeof(float));

  free(rgba);
  return image;
}
#endif

#if TSD_USE_OIIO
// stb decodes LDR files to float through stbi_ldr_to_hdr_gamma(), a plain
// pow(x, 2.2) rather than the true sRGB EOTF; OpenImageIO hands back the raw
// normalized values, so the same curve is applied here to keep every texture
// path on one contract. stb takes an odd channel count to be all-color and an
// even one to end in alpha (stb_image.h: `if (comp & 1) n = comp; else
// n = comp-1`), which is what leaves alpha linear -- match that exactly, or a
// 2-channel grey+alpha image gets its alpha gamma-corrected.
void applyGamma22InPlace(float *texels, size_t numTexels, int numChannels)
{
  const int numColorChannels =
      (numChannels & 1) ? numChannels : numChannels - 1;
  for (size_t t = 0; t < numTexels; t++) {
    float *texel = texels + t * numChannels;
    for (int c = 0; c < numColorChannels; c++)
      texel[c] = std::pow(texel[c], 2.2f);
  }
}

// TIFF is the format the USD/MaterialX assets reach for that stb cannot decode.
// OpenImageIO covers it (and everything else it has a reader for) without TSD
// taking on a format-specific decoder.
DecodedImage decodeOiio(const std::string &path, ColorSpace colorSpace)
{
  // OpenImageIO premultiplies unassociated alpha into the colour channels by
  // default; stb never does. Ask for the file's own values so a TIFF with
  // alpha lands on the same contract as every other texture path.
  OIIO::ImageSpec config;
  config.attribute("oiio:UnassociatedAlpha", 1);

  // The returned unique_ptr's deleter closes the file on every exit path.
  auto input = OIIO::ImageInput::open(path, &config);
  if (!input) {
    logError("[decodeImage] failed to open image '%s': %s",
        path.c_str(),
        OIIO::geterror().c_str());
    return {};
  }

  const auto &spec = input->spec();
  const int width = spec.width;
  const int height = spec.height;
  const int numChannels = spec.nchannels;
  if (width < 1 || height < 1 || numChannels < 1 || numChannels > 4) {
    logWarning("[decodeImage] image '%s' with %i channels not imported",
        path.c_str(),
        numChannels);
    return {};
  }
  // stb only ever gamma-decodes integer input; a float or half TIFF already
  // carries linear values, so applying the curve to it would darken the image.
  const bool fileIsIntegral = !spec.format.is_floating_point();

  DecodedImage image;
  image.elementType = texelTypeForChannelCount(numChannels);
  image.width = size_t(width);
  image.height = size_t(height);
  // OpenImageIO's scanline order runs from the picture's top.
  image.rowOrder = RowOrder::TOP_DOWN;
  image.texels.resize(
      size_t(width) * size_t(height) * size_t(numChannels) * sizeof(float));

  auto *texels = reinterpret_cast<float *>(image.texels.data());
  if (!input->read_image(0, 0, 0, numChannels, OIIO::TypeDesc::FLOAT, texels)) {
    logError("[decodeImage] failed to decode image '%s': %s",
        path.c_str(),
        input->geterror().c_str());
    return {};
  }

  if (colorSpace == ColorSpace::SRGB && fileIsIntegral) {
    applyGamma22InPlace(texels, size_t(width) * size_t(height), numChannels);
  }

  return image;
}
#endif

std::vector<char> readWholeFile(const std::string &path)
{
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    logError("[decodeImage] failed to open image '%s'", path.c_str());
    return {};
  }
  return std::vector<char>(
      (std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

} // namespace

ColorSpace colorSpaceForFile(const std::string &path, ColorSpace requested)
{
  const auto ext = lowered(extensionOf(path));
  if (ext == ".exr" || ext == ".dds")
    return ColorSpace::LINEAR;
  return requested;
}

ColorSpace colorSpaceForFormatHint(
    const std::string &formatHint, ColorSpace requested)
{
  return lowered(formatHint) == "dds" ? ColorSpace::LINEAR : requested;
}

DecodedImage decodeImageFile(const std::string &path, ColorSpace colorSpace)
{
  const auto ext = lowered(extensionOf(path));

  if (ext == ".dds") {
    const auto bytes = readWholeFile(path);
    return bytes.empty() ? DecodedImage{}
                         : decodeDds(bytes.data(), bytes.size(), path.c_str());
  }

#ifndef _WIN32
  if (ext == ".exr")
    return decodeExr(path);
#endif

  if (ext == ".tif" || ext == ".tiff") {
#if TSD_USE_OIIO
    return decodeOiio(path, colorSpace);
#else
    // Falling through to stb would fail with a decode error that says nothing
    // about the actual cause, which is that TSD was built without OpenImageIO.
    logError(
        "[decodeImage] cannot decode TIFF image '%s': TSD was built without"
        " OpenImageIO (set TSD_USE_OIIO=ON)",
        path.c_str());
    return {};
#endif
  }

  const auto bytes = readWholeFile(path);
  return bytes.empty()
      ? DecodedImage{}
      : decodeStb(bytes.data(), bytes.size(), colorSpace, path.c_str());
}

DecodedImage decodeImageFromMemory(const void *data,
    size_t numBytes,
    ColorSpace colorSpace,
    const std::string &formatHint,
    const std::string &id)
{
  if (lowered(formatHint) == "dds")
    return decodeDds(data, numBytes, id.c_str());
  return decodeStb(data, numBytes, colorSpace, id.c_str());
}

} // namespace tsd::io::detail
