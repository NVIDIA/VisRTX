// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Object.hpp"

namespace tsd {

struct Volume : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_VOLUME;

  DECLARE_OBJECT_DEFAULT_LIFETIME(Volume);

  Volume(Token subtype = tokens::unknown);
  virtual ~Volume() = default;

  anari::Object makeANARIObject(anari::Device d) const override;
};

namespace tokens::volume {

extern const Token transferFunction1D;

} // namespace tokens::volume

} // namespace tsd
