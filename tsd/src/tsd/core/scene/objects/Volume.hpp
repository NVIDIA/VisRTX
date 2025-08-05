// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Object.hpp"

namespace tsd::core {

struct Volume : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_VOLUME;

  DECLARE_OBJECT_DEFAULT_LIFETIME(Volume);

  Volume(Token subtype = tokens::unknown);
  virtual ~Volume() = default;

  anari::Object makeANARIObject(anari::Device d) const override;
};

using VolumeRef = IndexedVectorRef<Volume>;

namespace tokens::volume {

extern const Token transferFunction1D;

} // namespace tokens::volume

} // namespace tsd::core
