// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/objects/Light.hpp"
#include "tsd/core/scene/Scene.hpp"

namespace tsd::core {

Light::Light(Token subtype) : Object(ANARI_LIGHT, subtype)
{
  addParameter("visible").setValue(true).setDescription(
      "light is seen in primary visibility");

  if (subtype == tokens::light::hdri) {
    addParameter("up")
        .setValue(float3(0.f, 1.f, 0.f)) // NOTE: match viewport 'up' default
        .setDescription("up direction of the light in world-space");
    addParameter("direction")
        .setValue(float3(1.f, 0.f, 0.f))
        .setDescription(
            "direction to which the center of the texture will be mapped to");
    addParameter("scale")
        .setValue(1.f)
        .setDescription("scale factor for radiance")
        .setMin(0.f);
    return;
  }

  addParameter("color")
      .setValue(float3(1.f, 1.f, 1.f))
      .setDescription("light color")
      .setUsage(ParameterUsageHint::COLOR);

  if (subtype == tokens::light::directional) {
    addParameter("direction")
        .setValue(float2(0.f, 0.f))
        .setDescription("azimuth/elevation in degrees")
        .setUsage(ParameterUsageHint::DIRECTION)
        .setMin(float2(0.f, 0.f))
        .setMax(float2(360.f, 360.f));
    addParameter("irradiance")
        .setValue(1.f)
        .setDescription("light intensity on an illuminated surface")
        .setMin(0.f);
  } else if (subtype == tokens::light::point) {
    addParameter("position")
        .setValue(float3(0.f, 0.f, 0.f))
        .setDescription("the position of the point light");
    addParameter("radius")
        .setValue(0.f)
        .setDescription(
            "the radius of the sphere that emits light, in case of an arealight.")
        .setMin(0.f);
    addParameter("intensity")
        .setValue(1.f)
        .setDescription(
            "the overall amount of light emitted by the "
            "light in a direction, in W/sr")
        .setMin(0.f);
  } else if (subtype == tokens::light::quad) {
    addParameter("position")
        .setValue(float3(0.f, 0.f, 0.f))
        .setDescription("vector position");
    addParameter("edge1")
        .setValue(float3(1.f, 0.f, 0.f))
        .setDescription("vector to one adjacent vertex");
    addParameter("edge2")
        .setValue(float3(0.f, 1.f, 0.f))
        .setDescription("vector to the other adjacent vertex");
    addParameter("intensity")
        .setValue(1.f)
        .setDescription(
            "overall amount of light emitted "
            "in a direction, in W/sr")
        .setMin(0.f);
    addParameter("side")
        .setValue("front")
        .setDescription("which side of the rectangle emits light")
        .setStringValues({"front", "back", "both"});
  } else if (subtype == tokens::light::spot) {
    addParameter("position")
        .setValue(float3(0.f, 0.f, 0.f))
        .setDescription("the position of the spot light");
    addParameter("direction")
        .setValue(float3(0.f, 0.f, -1.f))
        .setDescription("main emission direction");
    addParameter("openingAngle")
        .setValue(float(M_PI))
        .setDescription("full opening angle (in radians) of the spot");
    addParameter("falloffAngle")
        .setValue(0.1f)
        .setDescription("size (angle in radians) between rim and spot");
    addParameter("intensity")
        .setValue(1.f)
        .setDescription(
            "the overall amount of light emitted by the "
            "light in a direction, in W/sr")
        .setMin(0.f);
  } else if (subtype == tokens::light::ring) {
    addParameter("position")
        .setValue(float3(0.f, 0.f, 0.f))
        .setDescription("the center of the ring light");
    addParameter("direction")
        .setValue(float3(0.f, 0.f, -1.f))
        .setDescription("main emission direction, the center axis of the ring");
    addParameter("openingAngle")
        .setValue(float(M_PI))
        .setDescription("full opening angle (in radians) of the cone of directions");
    addParameter("falloffAngle")
        .setValue(0.1f)
        .setDescription("size (angle in radians) of the region between the rim and full intensity");
    addParameter("radius")
        .setValue(0.f)
        .setDescription("the (outer) size of the ring, the radius of a disk with normal direction")
        .setMin(0.f);
    addParameter("innerRadius")
        .setValue(0.f)
        .setDescription("in combination with radius turns the disk into a ring; must be smaller than radius")
        .setMin(0.f);
    addParameter("intensity")
        .setValue(1.f)
        .setDescription(
            "the overall amount of light emitted by the "
            "light in a direction, in W/sr")
        .setMin(0.f);
  }
}

IndexedVectorRef<Light> Light::self() const
{
  return scene() ? scene()->getObject<Light>(index())
                 : IndexedVectorRef<Light>{};
}

anari::Object Light::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::Light>(d, subtype().c_str());
}

namespace tokens::light {

const Token directional = "directional";
const Token hdri = "hdri";
const Token point = "point";
const Token quad = "quad";
const Token ring = "ring";
const Token spot = "spot";

} // namespace tokens::light

} // namespace tsd::core
