// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Token.hpp"
#include "tsd/core/TypeMacros.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>

namespace tsd::io {

namespace detail {
struct DecodedImage;
} // namespace detail

// How a file's values relate to the linear values a renderer wants. Files that
// carry an encoding of their own (EXR, DDS) ignore this.
enum class ColorSpace
{
  SRGB,
  LINEAR
};

// Whether row 0 is the picture's top row or its bottom one. Decoders declare
// the order they produced; an ImageSource asks for the order it needs.
enum class RowOrder
{
  TOP_DOWN,
  BOTTOM_UP
};

// Identifies texel content -- not the sampler built from it. Two materials
// binding the same file at the same color space share one Image.
struct ImageSource
{
  // A resolved absolute path for file-backed images, and an importer-scoped
  // stable string otherwise ("gltf:<file>:image<N>",
  // "assimp://embedded/<N>", "pbrt:<file>::normal").
  std::string id;
  ColorSpace colorSpace{ColorSpace::SRGB};
  // The row order to store the image in, which is part of what identifies it:
  // the two orders are different bytes. Sampled images take the default,
  // because ANARI addresses texture coordinate (0, 0) at the image's top-left.
  // See docs/adr/0014-store-images-in-anari-orientation.md.
  RowOrder rowOrder{RowOrder::TOP_DOWN};
};

// A decoded image resident in a Scene.
struct Image
{
  tsd::scene::ArrayRef texels;
  // The picture's dimensions. Kept here rather than read back off the Array,
  // whose shape does not carry them for block-compressed texels: those are
  // the authored block stream rather than a texel grid.
  size_t width{0};
  size_t height{0};
  // The ANARI block format ("BC1_RGB", "BC7_SRGB", ...) the texels are the
  // block stream of; empty for a texel grid. A decoder that recognizes no
  // format yields no image at all, so this doubles as "is block-compressed".
  tsd::core::Token compressedFormat;
  // Set when the texels could not be brought into the order the source asked
  // for -- only block-compressed ones, which cannot be reordered without
  // decoding and re-encoding -- so makeImageSampler compensates in the
  // sampler's uv transform instead.
  bool vFlipInSampler{false};

  bool blockCompressed() const;

  explicit operator bool() const;
};

// Owns decoded images for one Scene. Holds the Scene it caches for so a cached
// ArrayRef can never reach a different Scene; it must not outlive that Scene.
class ImageCache
{
 public:
  ImageCache() = default;
  explicit ImageCache(tsd::scene::Scene *scene);

  // Copyable and moveable: an ImageCache is a value the caller owns, and
  // ImportContext holds one by value.
  TSD_DEFAULT_COPYABLE(ImageCache)
  TSD_DEFAULT_MOVEABLE(ImageCache)

  tsd::scene::Scene *scene() const;

  // Decode `source.id` as a file path.
  Image acquire(const ImageSource &source);
  // Decode an encoded image already in memory. `formatHint` names the
  // container ("dds", "png", ...) when the caller knows it; when it is empty
  // the decoder sniffs the bytes.
  Image acquire(const ImageSource &source,
      const void *data,
      size_t numBytes,
      const std::string &formatHint = "");
  // Adopt texels a caller decoded itself, declaring the row order they are in.
  Image acquireDecoded(const ImageSource &source,
      anari::DataType elementType,
      size_t width,
      size_t height,
      RowOrder rowOrder,
      const void *texels);

  // The image already held for `source`, or an invalid Image. For callers
  // that synthesize texels expensively and want to skip the work on a hit.
  Image find(const ImageSource &source) const;

  void clear();
  size_t size() const;

 private:
  Image store(const ImageSource &source, detail::DecodedImage &&decoded);

  tsd::scene::Scene *m_scene{nullptr};
  std::unordered_map<std::string, Image> m_images;
};

// An importer's own uv transform, in the form ANARI takes it.
struct UvTransform
{
  tsd::math::mat4 transform{tsd::math::IDENTITY_MAT4};
  tsd::math::float4 offset{0.f, 0.f, 0.f, 0.f};
};

// How a sampler reads the image it is bound to. Everything a binding can vary
// lives here, including the importer's own uv transform: `makeImageSampler`
// owns the sampler's `inTransform`/`inOffset` outright, because an image that
// could not be reordered needs a v-flip composed into them and a caller that
// set them afterwards would silently drop it.
struct SamplerSettings
{
  const char *inAttribute{"attribute0"};
  const char *wrapMode1{"repeat"};
  const char *wrapMode2{"repeat"};
  const char *filter{"linear"};
  // The importer's own uv transform, in the same form ANARI takes it. Unset
  // where the importer authored none, so a sampler that wants no transform is
  // left without the parameters entirely rather than with an identity.
  std::optional<UvTransform> uvTransform;
};

// Build a Sampler for an image the given cache produced. The cache names the
// Scene the Sampler lands in, so a caller cannot pair an image with a Scene it
// never reached. A cache with no Scene has no valid image to sample either, so
// this yields nothing rather than reaching for one.
tsd::scene::SamplerRef makeImageSampler(ImageCache &cache,
    const Image &image,
    const std::string &displayName,
    const SamplerSettings &settings = {});

// Inlined definitions ////////////////////////////////////////////////////////

inline bool Image::blockCompressed() const
{
  return !compressedFormat.empty();
}

inline Image::operator bool() const
{
  return texels.valid();
}

} // namespace tsd::io
