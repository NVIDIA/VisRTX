// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/objects/Geometry.hpp"
#include "tsd/core/scene/objects/Material.hpp"

namespace tsd::core {

struct Surface : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_SURFACE;

  DECLARE_OBJECT_DEFAULT_LIFETIME(Surface);

  Surface();
  virtual ~Surface() = default;

  void setGeometry(GeometryRef g);
  void setMaterial(MaterialRef m);

  anari::Object makeANARIObject(anari::Device d) const override;
};

using SurfaceRef = IndexedVectorRef<Surface>;

namespace tokens::surface {

extern const Token geometry;
extern const Token material;

} // namespace tokens::surface

} // namespace tsd
