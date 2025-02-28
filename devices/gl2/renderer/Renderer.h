// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Object.h"
// std
#include <limits>

namespace visgl2 {

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
