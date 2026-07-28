// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/scene/Scene.hpp"
// std
#include <cstddef>
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

// The row order a decoder produced. Declared by decoders, never by importers.
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
};

// A decoded image resident in a Scene.
struct Image
{
  tsd::scene::ArrayRef texels;
  // Block-compressed texels are the authored block stream rather than a
  // texel grid, so they cannot be reordered; makeImageSampler compensates.
  bool blockCompressed{false};

  explicit operator bool() const
  {
    return texels.valid();
  }
};

// Owns decoded images for one Scene. Holds the Scene it caches for so a cached
// ArrayRef can never reach a different Scene; it must not outlive that Scene.
class ImageCache
{
 public:
  ImageCache() = default;
  explicit ImageCache(tsd::scene::Scene *scene);

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
  Image *lookup(const ImageSource &source);
  Image store(const ImageSource &source, detail::DecodedImage &&decoded);

  tsd::scene::Scene *m_scene{nullptr};
  std::unordered_map<std::string, Image> m_images;
};

// How a sampler reads the image it is bound to. Everything a binding can vary
// lives here, including the importer's own uv transform: `makeImageSampler`
// owns the sampler's `inTransform`/`inOffset` outright, because a
// block-compressed image needs a v-flip composed into them and a caller that
// set them afterwards would silently drop it.
struct SamplerSettings
{
  const char *inAttribute{"attribute0"};
  const char *wrapMode1{"repeat"};
  const char *wrapMode2{"repeat"};
  const char *filter{"linear"};
  // The importer's own uv transform, in the same form ANARI takes it.
  tsd::math::mat4 uvTransform{tsd::math::IDENTITY_MAT4};
  tsd::math::float4 uvOffset{0.f, 0.f, 0.f, 0.f};
  // Set only when the importer authored a transform, so a sampler that wants
  // none is left without the parameters entirely rather than with an identity.
  bool hasUvTransform{false};
};

tsd::scene::SamplerRef makeImageSampler(tsd::scene::Scene &scene,
    const Image &image,
    const std::string &displayName,
    const SamplerSettings &settings = {});

} // namespace tsd::io
