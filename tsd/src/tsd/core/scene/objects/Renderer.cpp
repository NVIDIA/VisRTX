// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/objects/Renderer.hpp"
#include "tsd/core/scene/Scene.hpp"

namespace tsd::core {

Renderer::Renderer(Token sourceDevice, Token subtype)
    : Object(ANARI_RENDERER, subtype)
{
  m_rendererDeviceName = sourceDevice;

  addParameter("background")
      .setValue(float4(0.05f, 0.05f, 0.05f, 1.f))
      .setDescription("background color")
      .setUsage(ParameterUsageHint::COLOR);
  addParameter("ambientRadiance")
      .setValue(0.25f)
      .setDescription("intensity of ambient light")
      .setMin(0.f);
  addParameter("ambientColor")
      .setValue(float3(1.f))
      .setDescription("color of ambient light")
      .setUsage(ParameterUsageHint::COLOR);
}

ObjectPoolRef<Renderer> Renderer::self() const
{
  return scene() ? scene()->getObject<Renderer>(index())
                 : ObjectPoolRef<Renderer>{};
}

anari::Object Renderer::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::Renderer>(d, subtype().c_str());
}

} // namespace tsd::core
