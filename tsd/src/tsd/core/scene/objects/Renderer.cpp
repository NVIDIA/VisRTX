// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/objects/Renderer.hpp"
#include "tsd/core/scene/Scene.hpp"

namespace tsd::core {

Renderer::Renderer(Token sourceDevice, Token subtype)
    : Object(ANARI_RENDERER, subtype)
{
  m_rendererDeviceName = sourceDevice;
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
