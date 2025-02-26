// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "Renderer.h"

namespace visgl2 {

Renderer::Renderer(VisGL2DeviceGlobalState *s) : Object(ANARI_RENDERER, s) {}

Renderer *Renderer::createInstance(
    std::string_view /* subtype */, VisGL2DeviceGlobalState *s)
{
  return new Renderer(s);
}

void Renderer::commitParameters()
{
  m_background = getParam<vec4>("background", vec4(vec3(0.f), 1.f));
}

vec4 Renderer::background() const
{
  return m_background;
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_DEFINITION(visgl2::Renderer *);
