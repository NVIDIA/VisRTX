// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/objects/Material.hpp"
// std
#include <limits>

namespace tsd::core {

Material::Material(Token subtype) : Object(ANARI_MATERIAL, subtype)
{
  auto injectAlphaMode = [&]() {
    addParameter("alphaCutoff")
        .setValue(0.5f)
        .setDescription("threshold when alphaMode is 'mask'");
    addParameter("alphaMode")
        .setValue("blend")
        .setDescription("ANARI mode controlling opacity")
        .setStringValues({"opaque", "mask", "blend"})
        .setStringSelection(2);
  };

  if (subtype == tokens::material::matte) {
    addParameter("color")
        .setValue(float3(1.f, 0.f, 0.f))
        .setDescription("material color")
        .setUsage(ParameterUsageHint::COLOR);
    addParameter("opacity")
        .setValue(1.f)
        .setDescription("material opacity")
        .setMin(0.f)
        .setMax(1.f);
    injectAlphaMode();
  } else if (subtype == tokens::material::physicallyBased) {
    addParameter("baseColor")
        .setValue(float3(0.8f, 0.8f, 0.8f))
        .setDescription("base color")
        .setUsage(ParameterUsageHint::COLOR);
    addParameter("opacity")
        .setValue(1.f)
        .setDescription("opacity")
        .setMin(0.f)
        .setMax(1.f);
    injectAlphaMode();
    addParameter("metallic")
        .setValue(1.f)
        .setDescription("metalness")
        .setMin(0.f)
        .setMax(1.f);
    addParameter("roughness")
        .setValue(1.f)
        .setDescription("roughness")
        .setMin(0.f)
        .setMax(1.f);
    addParameter("emissive")
        .setValue(float3(0.f, 0.f, 0.f))
        .setDescription("strength of emissiveness")
        .setUsage(ParameterUsageHint::COLOR);
    addParameter("specular")
        .setValue(0.f)
        .setDescription("strength of the specular reflection")
        .setMin(0.f)
        .setMax(10.f);
    addParameter("specularColor")
        .setValue(float3(1.f, 1.f, 1.f))
        .setDescription("color of the specular reflection at normal incidence")
        .setUsage(ParameterUsageHint::COLOR);
    addParameter("clearcoat")
        .setValue(0.f)
        .setDescription("strength of the clearcoat layer")
        .setMin(0.f)
        .setMax(1.f);
    addParameter("clearcoatRoughness")
        .setValue(0.f)
        .setDescription("roughness of the clearcoat layer")
        .setMin(0.f)
        .setMax(1.f);
    addParameter("transmission")
        .setValue(0.f)
        .setDescription("strength of the transmission")
        .setMin(0.f)
        .setMax(1.f);
    addParameter("ior")
        .setValue(1.5f)
        .setDescription("index of refraction")
        .setMin(1.f)
        .setMax(4.f);
    addParameter("thickness")
        .setValue(0.f)
        .setDescription(
            "thickness of the volume beneath the surface "
            "(with 0 the material is thin-walled)")
        .setMin(0.f);
    addParameter("attenuationDistance")
        .setValue(1e20f)
        .setDescription(
            "average distance that light travels in the medium "
            "before interacting with a particle")
        .setMin(0.f);
    addParameter("attenuationColor")
        .setValue(float3(1.f, 1.f, 1.f))
        .setDescription(
            "color that white light turns into due to absorption "
            "when reaching the attenuation distance")
        .setUsage(ParameterUsageHint::COLOR);
    addParameter("sheenColor")
        .setValue(float3(0.f, 0.f, 0.f))
        .setDescription("sheen color")
        .setUsage(ParameterUsageHint::COLOR);
    addParameter("sheenRoughness")
        .setValue(0.f)
        .setDescription("sheen roughness")
        .setMin(0.f)
        .setMax(1.f);
    addParameter("iridescence")
        .setValue(0.f)
        .setDescription("stength of the thin-film layer");
    addParameter("iridescenceIor")
        .setValue(1.3f)
        .setDescription("index of refraction of the thin-film layer")
        .setMin(1.f)
        .setMax(4.f);
    addParameter("iridescenceThickness")
        .setValue(0.f)
        .setDescription("thickness of the thin-film layer")
        .setMin(0.f);
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

} // namespace tsd::core
