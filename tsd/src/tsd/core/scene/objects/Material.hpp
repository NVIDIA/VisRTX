// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Object.hpp"

namespace tsd::core {

struct Material : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_MATERIAL;

  DECLARE_OBJECT_DEFAULT_LIFETIME(Material);

  Material(Token subtype = tokens::unknown);
  virtual ~Material() = default;

  anari::Object makeANARIObject(anari::Device d) const override;
};

using MaterialRef = IndexedVectorRef<Material>;

namespace tokens::material {

extern const Token matte;
extern const Token physicallyBased;

} // namespace tokens::material

} // namespace tsd::core
