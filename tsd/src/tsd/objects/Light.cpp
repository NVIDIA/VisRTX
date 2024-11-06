// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/objects/Light.hpp"

namespace tsd {

Light::Light(Token subtype) : Object(ANARI_LIGHT, subtype)
{
  addParameter("visible"_t,
      true,
      "light is seen in primary visibility",
      tsd::ParameterUsageHint::NONE);

  if (subtype == tokens::light::hdri) {
    addParameter("up"_t,
        float3(0.f, 1.f, 0.f), // NOTE: match viewport 'up' default
        "up direction of the light in world-space",
        tsd::ParameterUsageHint::NONE);
    addParameter("direction"_t,
        float3(1.f, 0.f, 0.f),
        "direction to which the center of the texture will be mapped to",
        tsd::ParameterUsageHint::NONE);
    addParameter("scale"_t,
        1.f,
        "scale factor for radiance",
        tsd::ParameterUsageHint::NONE,
        0.f);
    return;
  }

  addParameter("color"_t,
      float3(1.f, 1.f, 1.f),
      "light color",
      ParameterUsageHint::COLOR);

  if (subtype == tokens::light::directional) {
    addParameter("direction"_t,
        float2(0.f, 0.f),
        "azimuth/elevation in degrees",
        tsd::ParameterUsageHint::DIRECTION,
        float2(0.f, 0.f),
        float2(360.f, 360.f));
    addParameter("irradiance"_t,
        1.f,
        "light intensity on an illuminated surface",
        tsd::ParameterUsageHint::NONE,
        0.f);
  } else if (subtype == tokens::light::point) {
    addParameter("position"_t,
        float3(0.f, 0.f, 0.f),
        "the position of the point light",
        tsd::ParameterUsageHint::NONE);
    addParameter("intensity"_t,
        1.f,
        "the overall amount of light emitted by the "
        "light in a direction, in W/sr",
        tsd::ParameterUsageHint::NONE,
        0.f);
  } else if (subtype == tokens::light::quad) {
    addParameter("position"_t,
        float3(0.f, 0.f, 0.f),
        "vector position",
        tsd::ParameterUsageHint::NONE);
    addParameter("edge1"_t,
        float3(1.f, 0.f, 0.f),
        "vector to one adjacent vertex",
        tsd::ParameterUsageHint::NONE);
    addParameter("edge2"_t,
        float3(0.f, 1.f, 0.f),
        "vector to the other adjacent vertex",
        tsd::ParameterUsageHint::NONE);
    addParameter("intensity"_t,
        1.f,
        "overall amount of light emitted in a direction, in W/sr",
        tsd::ParameterUsageHint::NONE,
        0.f);
  }
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

} // namespace tsd
