// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/objects/Geometry.hpp"
#include "tsd/objects/Material.hpp"

namespace tsd {

struct Surface : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_SURFACE;

  DECLARE_OBJECT_DEFAULT_LIFETIME(Surface);

  Surface();
  virtual ~Surface() = default;

  void setGeometry(IndexedVectorRef<Geometry> g);
  void setMaterial(IndexedVectorRef<Material> m);

  anari::Object makeANARIObject(anari::Device d) const override;
};

namespace tokens::surface {

extern const Token geometry;
extern const Token material;

} // namespace tokens::surface

} // namespace tsd
