// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/objects/Geometry.hpp"

namespace tsd {

Geometry::Geometry(Token stype) : Object(ANARI_GEOMETRY, stype) {}

anari::Object Geometry::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::Geometry>(d, subtype().c_str());
}

namespace tokens::geometry {

const Token cone = "cone";
const Token curve = "curve";
const Token cylinder = "cylinder";
const Token isosurface = "isosurface";
const Token quad = "quad";
const Token sphere = "sphere";
const Token triangle = "triangle";

} // namespace tokens::geometry

} // namespace tsd
