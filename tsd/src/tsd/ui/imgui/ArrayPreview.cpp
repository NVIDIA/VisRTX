// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ArrayPreview.h"
// imgui
#include <imgui.h>
// SDL3
#include <SDL3/SDL.h>
// std
#include <algorithm>
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

RGBA sample(const void *base, anari::DataType t, size_t i)
{
  switch (t) {
  case ANARI_FLOAT32: {
    auto *p = (const float *)base + i;
    return {p[0], 0.f, 0.f, 1.f};
  }
  case ANARI_FLOAT32_VEC2: {
    auto *p = (const float *)base + i * 2;
    return {p[0], p[1], 0.f, 1.f};
  }
  case ANARI_FLOAT32_VEC3: {
    auto *p = (const float *)base + i * 3;
    return {p[0], p[1], p[2], 1.f};
  }
  case ANARI_FLOAT32_VEC4: {
    auto *p = (const float *)base + i * 4;
    return {p[0], p[1], p[2], p[3]};
  }
  case ANARI_UFIXED8: {
    auto *p = (const uint8_t *)base + i;
    return {p[0] / 255.f, 0.f, 0.f, 1.f};
  }
  case ANARI_UFIXED8_VEC2: {
    auto *p = (const uint8_t *)base + i * 2;
    return {p[0] / 255.f, p[1] / 255.f, 0.f, 1.f};
  }
  case ANARI_UFIXED8_VEC3: {
    auto *p = (const uint8_t *)base + i * 3;
    return {p[0] / 255.f, p[1] / 255.f, p[2] / 255.f, 1.f};
  }
  case ANARI_UFIXED8_VEC4: {
    auto *p = (const uint8_t *)base + i * 4;
    return {p[0] / 255.f, p[1] / 255.f, p[2] / 255.f, p[3] / 255.f};
  }
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
