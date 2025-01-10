// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Object.hpp"

namespace tsd {

struct Sampler : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_SAMPLER;

  DECLARE_OBJECT_DEFAULT_LIFETIME(Sampler);

  Sampler(Token subtype = tokens::unknown);
  virtual ~Sampler() = default;

  anari::Object makeANARIObject(anari::Device d) const override;
};

using SamplerRef = IndexedVectorRef<Sampler>;

namespace tokens::sampler {

extern const Token image1D;
extern const Token image2D;
extern const Token image3D;
extern const Token primitive;
extern const Token transform;

} // namespace tokens::sampler

} // namespace tsd
