// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Object.hpp"

namespace tsd::core {

struct Light : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_LIGHT;

  DECLARE_OBJECT_DEFAULT_LIFETIME(Light);

  Light(Token subtype = tokens::unknown);
  virtual ~Light() = default;

  anari::Object makeANARIObject(anari::Device d) const override;
};

using LightRef = IndexedVectorRef<Light>;

namespace tokens::light {

extern const Token directional;
extern const Token hdri;
extern const Token point;
extern const Token quad;
extern const Token ring;
extern const Token spot;

} // namespace tokens::light

} // namespace tsd::core
