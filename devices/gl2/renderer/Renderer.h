// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Object.h"
// std
#include <limits>

namespace visgl2 {

// This is just an example struct of what data is associated with each pixel.
struct PixelSample
{
  vec4 color{0.f, 1.f, 0.f, 1.f};
  float depth{std::numeric_limits<float>::max()};

  PixelSample(vec4 c) : color(c) {}
};

// Inherit from this, add your functionality, and add it to 'createInstance()'
struct Renderer : public Object
{
  Renderer(VisGL2DeviceGlobalState *s);
  ~Renderer() override = default;

  static Renderer *createInstance(
      std::string_view subtype, VisGL2DeviceGlobalState *d);

  void commitParameters() override;

  vec4 background() const;

 private:
  vec4 m_background{0.f, 0.f, 0.f, 1.f};
};

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_SPECIALIZATION(visgl2::Renderer *, ANARI_RENDERER);
