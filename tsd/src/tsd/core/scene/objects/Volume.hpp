// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Object.hpp"

namespace tsd::core {

struct Volume : public Object
{
  DECLARE_OBJECT_DEFAULT_LIFETIME(Volume);

  Volume(Token subtype = tokens::unknown);
  virtual ~Volume() = default;

  ObjectPoolRef<Volume> self() const;

  anari::Object makeANARIObject(anari::Device d) const override;
};

using VolumeRef = ObjectPoolRef<Volume>;

namespace tokens::volume {

extern const Token structuredRegular;
extern const Token structuredRectilinear;
extern const Token transferFunction1D;

} // namespace tokens::volume

} // namespace tsd::core
