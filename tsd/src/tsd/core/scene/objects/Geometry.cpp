// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/objects/Geometry.hpp"
#include "tsd/core/scene/Scene.hpp"

namespace tsd::core {

Geometry::Geometry(Token stype) : Object(ANARI_GEOMETRY, stype) {}

ObjectPoolRef<Geometry> Geometry::self() const
{
  return scene() ? scene()->getObject<Geometry>(index())
                 : ObjectPoolRef<Geometry>{};
}

anari::Object Geometry::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::Geometry>(d, subtype().c_str());
}

namespace tokens::geometry {

const Token cone = "cone";
const Token curve = "curve";
const Token cylinder = "cylinder";
const Token isosurface = "isosurface";
const Token neural = "neural";
const Token quad = "quad";
const Token sphere = "sphere";
const Token triangle = "triangle";

} // namespace tokens::geometry

} // namespace tsd::core
