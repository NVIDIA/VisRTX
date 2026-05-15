// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ArrayPreview.h"
// imgui
#include <imgui.h>
// SDL3
#include <SDL3/SDL.h>
// std
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace tsd::ui {

namespace {

constexpr int kStripHeight = 16;
constexpr int kMaxDim = 128;

struct Entry
{
  SDL_Texture *tex{nullptr};
  int w{0};
  int h{0};
};

struct Cache
{
  SDL_Renderer *renderer{nullptr};
  std::unordered_map<size_t, Entry> entries;

  ~Cache()
  {
    clear();
  }

  void clear()
  {
    for (auto &kv : entries)
      if (kv.second.tex)
        SDL_DestroyTexture(kv.second.tex);
    entries.clear();
  }
};

Cache &cache()
{
  static Cache c;
  return c;
}

// element-type → RGBA float sampling //
//
// Sampler-channel semantics: R → (r,0,0,1), RG → (r,g,0,1), RGB → (r,g,b,1),
// RGBA → (r,g,b,a). Missing channels are zero, missing alpha is opaque.

using RGBA = std::array<float, 4>;

template <typename T>
inline float toNormalizedFloat(T v);

template <>
inline float toNormalizedFloat<float>(float v)
{
  return v;
}
template <>
inline float toNormalizedFloat<uint8_t>(uint8_t v)
{
  return v / 255.f;
}
template <>
inline float toNormalizedFloat<uint16_t>(uint16_t v)
{
  return v / 65535.f;
}
template <>
inline float toNormalizedFloat<uint32_t>(uint32_t v)
{
  return v / 4294967295.f;
}
template <>
inline float toNormalizedFloat<int8_t>(int8_t v)
{
  return std::max(v / 127.f, -1.f);
}
template <>
inline float toNormalizedFloat<int16_t>(int16_t v)
{
  return std::max(v / 32767.f, -1.f);
}
template <>
inline float toNormalizedFloat<int32_t>(int32_t v)
{
  return std::max(v / 2147483647.f, -1.f);
}

// linear → sRGB (IEC 61966-2-1). HDR values are clamped to [0, 1] first, so
// previews show clipped highlights rather than wrapping or going black.
inline float linearToSrgb(float c)
{
  c = std::clamp(c, 0.f, 1.f);
  return c <= 0.0031308f ? c * 12.92f
                         : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
}

// Preview keeps every sample in display space.
//
// Integer formats: normalized to [0, 1] and passed through. sRGB-tagged
// formats deliberately skip the sRGB → linear decode — the texture is
// blitted without a matching linear → sRGB encode, so decoding would make
// sRGB-tagged previews look darker than identical content tagged as plain
// UFIXED8.
//
// Float formats (incl. HDRIs): treated as linear-light and encoded to sRGB
// here. Without this encode a linear value of 0.5 would display as if it
// were sRGB 0.5, i.e. visibly too dark.
template <typename T, int N>
inline RGBA sample(const void *base, size_t i)
{
  auto *p = (const T *)base + i * N;
  RGBA out = {0.f, 0.f, 0.f, 1.f};
  for (int c = 0; c < N; ++c)
    out[c] = toNormalizedFloat<T>(p[c]);
  if constexpr (std::is_floating_point_v<T>) {
    const int colorChans = (N == 4) ? 3 : N;
    for (int c = 0; c < colorChans; ++c)
      out[c] = linearToSrgb(out[c]);
  }
  return out;
}

RGBA sample(const void *base, anari::DataType t, size_t i)
{
  switch (t) {
  // Float
  case ANARI_FLOAT32:
    return sample<float, 1>(base, i);
  case ANARI_FLOAT32_VEC2:
    return sample<float, 2>(base, i);
  case ANARI_FLOAT32_VEC3:
    return sample<float, 3>(base, i);
  case ANARI_FLOAT32_VEC4:
    return sample<float, 4>(base, i);
  // UFIXED8
  case ANARI_UFIXED8:
    return sample<uint8_t, 1>(base, i);
  case ANARI_UFIXED8_VEC2:
    return sample<uint8_t, 2>(base, i);
  case ANARI_UFIXED8_VEC3:
    return sample<uint8_t, 3>(base, i);
  case ANARI_UFIXED8_VEC4:
    return sample<uint8_t, 4>(base, i);
  // UFIXED8 sRGB — sampled identically to plain UFIXED8 (see sample<>).
  case ANARI_UFIXED8_R_SRGB:
    return sample<uint8_t, 1>(base, i);
  case ANARI_UFIXED8_RA_SRGB:
    return sample<uint8_t, 2>(base, i);
  case ANARI_UFIXED8_RGB_SRGB:
    return sample<uint8_t, 3>(base, i);
  case ANARI_UFIXED8_RGBA_SRGB:
    return sample<uint8_t, 4>(base, i);
  // FIXED8 (signed normalized)
  case ANARI_FIXED8:
    return sample<int8_t, 1>(base, i);
  case ANARI_FIXED8_VEC2:
    return sample<int8_t, 2>(base, i);
  case ANARI_FIXED8_VEC3:
    return sample<int8_t, 3>(base, i);
  case ANARI_FIXED8_VEC4:
    return sample<int8_t, 4>(base, i);
  // UFIXED16
  case ANARI_UFIXED16:
    return sample<uint16_t, 1>(base, i);
  case ANARI_UFIXED16_VEC2:
    return sample<uint16_t, 2>(base, i);
  case ANARI_UFIXED16_VEC3:
    return sample<uint16_t, 3>(base, i);
  case ANARI_UFIXED16_VEC4:
    return sample<uint16_t, 4>(base, i);
  // FIXED16
  case ANARI_FIXED16:
    return sample<int16_t, 1>(base, i);
  case ANARI_FIXED16_VEC2:
    return sample<int16_t, 2>(base, i);
  case ANARI_FIXED16_VEC3:
    return sample<int16_t, 3>(base, i);
  case ANARI_FIXED16_VEC4:
    return sample<int16_t, 4>(base, i);
  // UFIXED32
  case ANARI_UFIXED32:
    return sample<uint32_t, 1>(base, i);
  case ANARI_UFIXED32_VEC2:
    return sample<uint32_t, 2>(base, i);
  case ANARI_UFIXED32_VEC3:
    return sample<uint32_t, 3>(base, i);
  case ANARI_UFIXED32_VEC4:
    return sample<uint32_t, 4>(base, i);
  // FIXED32
  case ANARI_FIXED32:
    return sample<int32_t, 1>(base, i);
  case ANARI_FIXED32_VEC2:
    return sample<int32_t, 2>(base, i);
  case ANARI_FIXED32_VEC3:
    return sample<int32_t, 3>(base, i);
  case ANARI_FIXED32_VEC4:
    return sample<int32_t, 4>(base, i);
  default:
    return {1.f, 0.f, 1.f, 1.f}; // magenta sentinel
  }
}

bool isPreviewableElement(anari::DataType t)
{
  switch (t) {
  case ANARI_FLOAT32:
  case ANARI_FLOAT32_VEC2:
  case ANARI_FLOAT32_VEC3:
  case ANARI_FLOAT32_VEC4:
  case ANARI_UFIXED8:
  case ANARI_UFIXED8_VEC2:
  case ANARI_UFIXED8_VEC3:
  case ANARI_UFIXED8_VEC4:
  case ANARI_UFIXED8_R_SRGB:
  case ANARI_UFIXED8_RA_SRGB:
  case ANARI_UFIXED8_RGB_SRGB:
  case ANARI_UFIXED8_RGBA_SRGB:
  case ANARI_FIXED8:
  case ANARI_FIXED8_VEC2:
  case ANARI_FIXED8_VEC3:
  case ANARI_FIXED8_VEC4:
  case ANARI_UFIXED16:
  case ANARI_UFIXED16_VEC2:
  case ANARI_UFIXED16_VEC3:
  case ANARI_UFIXED16_VEC4:
  case ANARI_FIXED16:
  case ANARI_FIXED16_VEC2:
  case ANARI_FIXED16_VEC3:
  case ANARI_FIXED16_VEC4:
  case ANARI_UFIXED32:
  case ANARI_UFIXED32_VEC2:
  case ANARI_UFIXED32_VEC3:
  case ANARI_UFIXED32_VEC4:
  case ANARI_FIXED32:
  case ANARI_FIXED32_VEC2:
  case ANARI_FIXED32_VEC3:
  case ANARI_FIXED32_VEC4:
    return true;
  default:
    return false;
  }
}

// Photoshop-style alpha checkerboard. 8 px cells, light/dark gray.
constexpr int kCheckerCell = 8;
constexpr float kCheckerLight = 0.75f;
constexpr float kCheckerDark = 0.50f;

inline RGBA composite(RGBA px, int x, int y)
{
  const bool light = (((x / kCheckerCell) + (y / kCheckerCell)) & 1) == 0;
  const float bg = light ? kCheckerLight : kCheckerDark;
  const float a = std::clamp(px[3], 0.f, 1.f);
  return {bg * (1.f - a) + px[0] * a,
      bg * (1.f - a) + px[1] * a,
      bg * (1.f - a) + px[2] * a,
      1.f};
}

// thumbnail builders //

SDL_Texture *makeTexture(
    SDL_Renderer *r, const std::vector<RGBA> &pixels, int w, int h)
{
  auto *tex = SDL_CreateTexture(
      r, SDL_PIXELFORMAT_RGBA128_FLOAT, SDL_TEXTUREACCESS_STATIC, w, h);
  if (!tex)
    return nullptr;
  SDL_UpdateTexture(tex, nullptr, pixels.data(), w * int(sizeof(RGBA)));
  SDL_SetTextureScaleMode(tex, SDL_SCALEMODE_LINEAR);
  return tex;
}

Entry build1DColor(SDL_Renderer *r, const tsd::scene::Array &a)
{
  const size_t n = a.dim(0);
  const auto t = a.elementType();
  const auto *base = a.data();
  if (n == 0 || !base)
    return {};

  const int w = std::min<int>(kMaxDim, int(n));
  std::vector<RGBA> row(w);
  for (int x = 0; x < w; ++x) {
    const size_t srcIdx = size_t(x) * n / size_t(w);
    row[x] = sample(base, t, srcIdx);
  }

  // Replicate row vertically for readable strip; composite over checkerboard
  // so partial-alpha samples (e.g. transfer function colormaps) read correctly.
  std::vector<RGBA> pixels(size_t(w) * kStripHeight);
  for (int y = 0; y < kStripHeight; ++y)
    for (int x = 0; x < w; ++x)
      pixels[size_t(y) * w + x] = composite(row[x], x, y);

  return {makeTexture(r, pixels, w, kStripHeight), w, kStripHeight};
}

Entry build2DColor(SDL_Renderer *r, const tsd::scene::Array &a)
{
  const size_t srcW = a.dim(0);
  const size_t srcH = a.dim(1);
  const auto t = a.elementType();
  const auto *base = a.data();
  if (srcW == 0 || srcH == 0 || !base)
    return {};

  // Aspect-preserve, cap longest axis at kMaxDim.
  const float scale = float(kMaxDim) / float(std::max(srcW, srcH));
  const int w = std::max(1, int(srcW * scale));
  const int h = std::max(1, int(srcH * scale));

  std::vector<RGBA> pixels(size_t(w) * h);
  for (int y = 0; y < h; ++y) {
    const size_t sy = size_t(y) * srcH / size_t(h);
    for (int x = 0; x < w; ++x) {
      const size_t sx = size_t(x) * srcW / size_t(w);
      pixels[size_t(y) * w + x] =
          composite(sample(base, t, sy * srcW + sx), x, y);
    }
  }

  return {makeTexture(r, pixels, w, h), w, h};
}

Entry buildEntry(SDL_Renderer *r, const tsd::scene::Array &a)
{
  if (!a.isHost() || a.isEmpty())
    return {};
  if (!isPreviewableElement(a.elementType()))
    return {};

  switch (a.type()) {
  case ANARI_ARRAY1D:
    return build1DColor(r, a);
  case ANARI_ARRAY2D:
    return build2DColor(r, a);
  default:
    return {};
  }
}

const Entry *getOrCreate(const tsd::scene::Array &a)
{
  auto &c = cache();
  if (!c.renderer)
    return nullptr;

  const size_t key = a.index();
  if (auto it = c.entries.find(key); it != c.entries.end())
    return &it->second;

  auto entry = buildEntry(c.renderer, a);
  if (!entry.tex)
    return nullptr;

  auto [it, _] = c.entries.emplace(key, entry);
  return &it->second;
}

} // namespace

void setupArrayPreview(SDL_Renderer *r)
{
  auto &c = cache();
  if (c.renderer == r)
    return;
  c.clear();
  c.renderer = r;
}

void teardownArrayPreview()
{
  cache().clear();
  cache().renderer = nullptr;
}

bool buildUI_array_preview(const tsd::scene::Array &a, float maxDim)
{
  const auto *e = getOrCreate(a);
  if (!e)
    return false;

  const float scale = maxDim / float(std::max(e->w, e->h));
  const ImVec2 size(e->w * scale, e->h * scale);
  ImGui::Image(reinterpret_cast<ImTextureID>(e->tex), size);
  return true;
}

} // namespace tsd::ui
