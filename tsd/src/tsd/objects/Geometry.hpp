// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Object.hpp"

namespace tsd {

struct Geometry : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_GEOMETRY;

  DECLARE_OBJECT_DEFAULT_LIFETIME(Geometry);

  Geometry(Token subtype = tokens::unknown);
  virtual ~Geometry() = default;

  anari::Object makeANARIObject(anari::Device d) const override;
};

using GeometryRef = IndexedVectorRef<Geometry>;

namespace tokens::geometry {

extern const Token cone;
extern const Token curve;
extern const Token cylinder;
extern const Token isosurface;
extern const Token quad;
extern const Token sphere;
extern const Token triangle;

} // namespace tokens::geometry

} // namespace tsd
