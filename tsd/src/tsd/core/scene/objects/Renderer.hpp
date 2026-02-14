// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Object.hpp"

namespace tsd::core {

struct Renderer : public Object
{
  DECLARE_OBJECT_DEFAULT_LIFETIME(Renderer);

  Renderer() = default;
  Renderer(Token sourceDevice, Token subtype);
  virtual ~Renderer() = default;

  ObjectPoolRef<Renderer> self() const;

  anari::Object makeANARIObject(anari::Device d) const override;
};

using RendererRef = ObjectPoolRef<Renderer>;

} // namespace tsd::core
