// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/objects/Material.hpp"
// std
#include <limits>

namespace tsd {

Material::Material(Token subtype) : Object(ANARI_MATERIAL, subtype)
{
  auto injectAlphaMode = [&]() {
    addParameter("alphaCutoff"_t, 0.5f, "threshold when alphaMode is 'mask'");
    auto &alphaMode =
        addParameter("alphaMode"_t, "blend", "ANARI mode controlling opacity");
    alphaMode.setStringValues({"opaque", "mask", "blend"});
    alphaMode.setStringSelection(2);
  };

  if (subtype == tokens::material::matte) {
    addParameter("color"_t,
        float3(1.f, 0.f, 0.f),
        "material color",
        ParameterUsageHint::COLOR);
    addParameter("opacity"_t,
        1.f,
        "material opacity",
        ParameterUsageHint::NONE,
        0.f,
        1.f);
    injectAlphaMode();
  } else if (subtype == tokens::material::physicallyBased) {
    addParameter("baseColor"_t,
        float3(0.8f, 0.8f, 0.8f),
        "base color",
        ParameterUsageHint::COLOR);
    addParameter(
        "opacity"_t, 1.f, "opacity", ParameterUsageHint::NONE, 0.f, 1.f);
    injectAlphaMode();
    addParameter(
        "metallic"_t, 1.f, "metalness", ParameterUsageHint::NONE, 0.f, 1.f);
    addParameter(
        "roughness"_t, 1.f, "roughness", ParameterUsageHint::NONE, 0.f, 1.f);
    addParameter("emissive"_t,
        float3(0.f, 0.f, 0.f),
        "emissive",
        ParameterUsageHint::COLOR);
    addParameter("specular"_t,
        0.f,
        "strength of the specular reflection",
        ParameterUsageHint::NONE,
        0.f,
        10.f);
    addParameter("specularColor"_t,
        float3(1.f, 1.f, 1.f),
        "color of the specular reflection at normal incidence",
        ParameterUsageHint::COLOR);
    addParameter("clearcoat"_t,
        0.f,
        "strength of the clearcoat layer",
        ParameterUsageHint::NONE,
        0.f,
        1.f);
    addParameter("clearcoatRoughness"_t,
        0.f,
        "roughness of the clearcoat layer",
        ParameterUsageHint::NONE,
        0.f,
        1.f);
    addParameter("transmission"_t,
        0.f,
        "strength of the transmission",
        ParameterUsageHint::NONE,
        0.f,
        1.f);
    addParameter("ior"_t,
        1.5f,
        "index of refraction",
        ParameterUsageHint::NONE,
        1.f,
        4.f);
    addParameter("thickness"_t,
        0.f,
        "thickness of the volume beneath the surface "
        "(with 0 the material is thin-walled)",
        ParameterUsageHint::NONE,
        0.f);
    addParameter("attenuationDistance"_t,
        std::numeric_limits<float>::max(),
        "average distance that light travels in the medium "
        "before interacting with a particle",
        ParameterUsageHint::NONE,
        0.f);
    addParameter("attenuationColor"_t,
        float3(1.f, 1.f, 1.f),
        "color that white light turns into due to absorption "
        "when reaching the attenuation distance",
        ParameterUsageHint::COLOR);
    addParameter("sheenColor"_t,
        float3(0.f, 0.f, 0.f),
        "sheen color",
        ParameterUsageHint::COLOR);
    addParameter("sheenRoughness"_t,
        0.f,
        "sheen roughness",
        ParameterUsageHint::NONE,
        0.f,
        1.f);
    addParameter("iridescence"_t,
        0.f,
        "stength of the thin-film layer",
        ParameterUsageHint::NONE);
    addParameter("iridescenceIor"_t,
        1.3f,
        "index of refraction of the thin-film layer",
        ParameterUsageHint::NONE,
        1.f,
        4.f);
    addParameter("iridescenceThickness"_t,
        0.f,
        "thickness of the thin-film layer",
        ParameterUsageHint::NONE,
        0.f);
  }
}

anari::Object Material::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::Material>(d, subtype().c_str());
}

namespace tokens::material {

Token const matte = "matte";
Token const physicallyBased = "physicallyBased";

} // namespace tokens::material

} // namespace tsd
