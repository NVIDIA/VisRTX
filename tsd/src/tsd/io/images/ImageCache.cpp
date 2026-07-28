// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/images/ImageCache.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/images/detail/decoders.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <algorithm>
#include <cstring>
#include <utility>

using U64Vec2 = tsd::math::vec<std::uint64_t, 2>;
namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(U64Vec2, ANARI_UINT64_VEC2);
}

namespace tsd::io {

using namespace tsd::core;
using namespace tsd::scene;

namespace {

// The row order a decoded image is stored in once it is resident in a Scene.
//
// This is what every decoder is normalized to and the only place in the tree
// that decides orientation. It is TOP_DOWN here because that is what all seven
// decode paths produced before this component existed; docs/adr/ records the
// move to ANARI orientation.
constexpr RowOrder SCENE_ROW_ORDER = RowOrder::TOP_DOWN;

std::string keyOf(const ImageSource &source)
{
  return source.id
      + (source.colorSpace == ColorSpace::LINEAR ? "_linear" : "_srgb");
}

// Reverse the image's rows in place so it lands in SCENE_ROW_ORDER.
void normalizeRowOrder(detail::DecodedImage &image)
{
  if (image.rowOrder == SCENE_ROW_ORDER || image.height < 2)
    return;

  if (image.blockCompressed) {
    // BC blocks cover 4x4 texels, so reversing rows would mean decoding and
    // re-encoding, which is the whole cost compressedImage2D exists to avoid.
    // makeImageSampler compensates in the sampler's transform instead.
    return;
  }

  const size_t rowBytes = image.texels.size() / image.height;
  auto *first = image.texels.data();
  for (size_t r = 0; r < image.height / 2; ++r) {
    std::swap_ranges(first + r * rowBytes,
        first + (r + 1) * rowBytes,
        first + (image.height - 1 - r) * rowBytes);
  }
  image.rowOrder = SCENE_ROW_ORDER;
}

} // namespace

ImageCache::ImageCache(Scene *scene) : m_scene(scene) {}

Scene *ImageCache::scene() const
{
  return m_scene;
}

Image ImageCache::acquire(const ImageSource &source)
{
  auto resolved = source;
  resolved.colorSpace =
      detail::colorSpaceForFile(source.id, source.colorSpace);

  if (auto *cached = lookup(resolved))
    return *cached;

  return store(resolved,
      detail::decodeImageFile(resolved.id, resolved.colorSpace));
}

Image ImageCache::acquire(const ImageSource &source,
    const void *data,
    size_t numBytes,
    const std::string &formatHint)
{
  auto resolved = source;
  resolved.colorSpace =
      detail::colorSpaceForFormatHint(formatHint, source.colorSpace);

  if (auto *cached = lookup(resolved))
    return *cached;

  return store(resolved,
      detail::decodeImageFromMemory(
          data, numBytes, resolved.colorSpace, formatHint, resolved.id));
}

Image ImageCache::acquireDecoded(const ImageSource &source,
    anari::DataType elementType,
    size_t width,
    size_t height,
    RowOrder rowOrder,
    const void *texels)
{
  if (auto *cached = lookup(source))
    return *cached;

  detail::DecodedImage decoded;
  decoded.elementType = elementType;
  decoded.width = width;
  decoded.height = height;
  decoded.rowOrder = rowOrder;
  const auto numBytes = width * height * anari::sizeOf(elementType);
  const auto *bytes = static_cast<const char *>(texels);
  decoded.texels.assign(bytes, bytes + numBytes);

  return store(source, std::move(decoded));
}

Image ImageCache::find(const ImageSource &source) const
{
  auto found = m_images.find(keyOf(source));
  return found == m_images.end() ? Image{} : found->second;
}

void ImageCache::clear()
{
  m_images.clear();
}

size_t ImageCache::size() const
{
  return m_images.size();
}

Image *ImageCache::lookup(const ImageSource &source)
{
  auto found = m_images.find(keyOf(source));
  return found == m_images.end() ? nullptr : &found->second;
}

Image ImageCache::store(
    const ImageSource &source, detail::DecodedImage &&decoded)
{
  if (!decoded)
    return {};

  if (!m_scene) {
    logError("[ImageCache] no scene to store image '%s' in", source.id.c_str());
    return {};
  }

  normalizeRowOrder(decoded);

  Image image;
  image.blockCompressed = decoded.blockCompressed;
  if (decoded.blockCompressed) {
    image.texels = m_scene->createArray(ANARI_INT8, decoded.texels.size());
    image.texels->setData(decoded.texels.data());
    image.texels->setMetadataValue(
        "compressedFormat", decoded.compressedFormat.value());
    image.texels->setMetadataValue(
        "imageSize", U64Vec2(decoded.width, decoded.height));
  } else {
    image.texels = m_scene->createArray(
        decoded.elementType, decoded.width, decoded.height);
    image.texels->setData(decoded.texels.data());
  }

  m_images[keyOf(source)] = image;
  return image;
}

SamplerRef makeImageSampler(Scene &scene,
    const Image &image,
    const std::string &displayName,
    const SamplerSettings &settings)
{
  if (!image)
    return {};

  auto sampler = scene.createObject<Sampler>(image.blockCompressed
          ? tokens::sampler::compressedImage2D
          : tokens::sampler::image2D);

  sampler->setParameterObject("image", *image.texels);
  if (image.blockCompressed) {
    const auto compressedFormat =
        image.texels->getMetadataValue("compressedFormat").getString();
    sampler->setParameter("format", compressedFormat.c_str());
    sampler->setParameter(
        "size", image.texels->getMetadataValue("imageSize").get<U64Vec2>());
  }
  sampler->setParameter("inAttribute", settings.inAttribute);
  sampler->setParameter("wrapMode1", settings.wrapMode1);
  sampler->setParameter("wrapMode2", settings.wrapMode2);
  sampler->setParameter("filter", settings.filter);
  if (settings.hasUvTransform) {
    sampler->setParameter("inTransform", settings.uvTransform);
    sampler->setParameter("inOffset", settings.uvOffset);
  }
  sampler->setName(fileOf(displayName).c_str());

  return sampler;
}

} // namespace tsd::io
