// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/objects/Surface.hpp"

namespace tsd {

Surface::Surface() : Object(ANARI_SURFACE) {}

void Surface::setGeometry(IndexedVectorRef<Geometry> g)
{
  setParameterObject(tokens::surface::geometry, *g);
}

void Surface::setMaterial(IndexedVectorRef<Material> m)
{
  setParameterObject(tokens::surface::material, *m);
}

anari::Object Surface::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::Surface>(d);
}

namespace tokens::surface {

const Token geometry = "geometry";
const Token material = "material";

} // namespace tokens::surface

} // namespace tsd
