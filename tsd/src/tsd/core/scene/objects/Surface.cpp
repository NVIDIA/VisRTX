// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/objects/Surface.hpp"
#include "tsd/core/scene/Scene.hpp"

namespace tsd::core {

Surface::Surface() : Object(ANARI_SURFACE)
{
  addParameter("visible").setValue(true).setDescription(
      "whether the volume is visible in the scene");
}

void Surface::setGeometry(GeometryRef g)
{
  setParameterObject(tokens::surface::geometry, *g);
}

void Surface::setMaterial(MaterialRef m)
{
  setParameterObject(tokens::surface::material, *m);
}

Geometry *Surface::geometry() const
{
  return parameterValueAsObject<Geometry>(tokens::surface::geometry);
}

Material *Surface::material() const
{
  return parameterValueAsObject<Material>(tokens::surface::material);
}

ObjectPoolRef<Surface> Surface::self() const
{
  return scene() ? scene()->getObject<Surface>(index())
                 : ObjectPoolRef<Surface>{};
}

anari::Object Surface::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::Surface>(d);
}

namespace tokens::surface {

const Token geometry = "geometry";
const Token material = "material";

} // namespace tokens::surface

} // namespace tsd::core
