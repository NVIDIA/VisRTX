// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/scene/objects/Material.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <limits>

namespace tsd::scene {

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
  } else if (subtype == tokens::material::physicallyBased || subtype == tokens::material::physicallyBasedMDL) {
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
    addParameter("anisotropyStrength")
        .setValue(0.f)
        .setDescription("anisotropy strength in [0:1]")
        .setMin(0.f)
        .setMax(1.f);
    addParameter("anisotropyDirection")
        .setValue(float2(1.f, 0.f))
        .setDescription(
            "direction in tangent and bitangent space, in [-1:1]^2");
    addParameter("anisotropyRotation")
        .setValue(0.f)
        .setDescription("anisotropy rotation (in radians)")
        .setMin(0.f)
        //.setMax(M_PI*2.f); /* not sure why, but this leads to crashes.... */
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
  } else if (subtype == tokens::material::mdl) {
    addParameter("source")
        .setValue("::visrtx::default::diffuseWhite")
        .setDescription("MDL module name");
  } else if (subtype == tokens::material::materialx) {
    addParameter("source")
        .setValue("")
        .setDescription("A .mtlx document: a file path (documentFile) or "
                        "inline .mtlx XML (documentInline)");
    addParameter("sourceType")
        .setValue("documentFile")
        .setDescription("How source is interpreted: documentFile or "
                        "documentInline");
    addParameter("materialName")
        .setValue("")
        .setDescription("Material to select within the document");
  }
}

ObjectPoolRef<Material> Material::self() const
{
  return scene() ? scene()->getObject<Material>(index())
                 : ObjectPoolRef<Material>{};
}

anari::Object Material::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::Material>(d, subtype().c_str());
}

// The instantiation is scene content (ADR 0008, devices/rtx/docs/adr): the
// standard_surface nodedef it binds comes from the MaterialX distribution the
// device resolves at runtime — nothing is embedded or shipped for this.
constexpr const char *kStandardSurfaceInstantiation =
    R"(<?xml version="1.0"?>
<materialx version="1.39">
  <standard_surface name="surface" type="surfaceshader" />
  <surfacematerial name="StandardSurface" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";

void applyMaterialXStandardSurfacePreset(Material &m)
{
  m.setParameter("sourceType", "documentInline");
  m.setParameter("source", kStandardSurfaceInstantiation);
  m.setParameter("materialName", "StandardSurface");

  auto addF = [&](const char *n, float v, float lo, float hi, const char *d) {
    m.addParameter(n).setValue(v).setMin(lo).setMax(hi).setDescription(d);
  };
  auto addColor = [&](const char *n, float3 v, const char *d) {
    m.addParameter(n)
        .setValue(v)
        .setUsage(ParameterUsageHint::COLOR)
        .setDescription(d);
  };

  addF("base", 1.f, 0.f, 1.f, "diffuse weight");
  addColor("base_color", float3(0.8f, 0.8f, 0.8f), "diffuse color");
  addF("metalness", 0.f, 0.f, 1.f, "metalness");
  addF("specular", 1.f, 0.f, 1.f, "specular weight");
  addColor("specular_color", float3(1.f, 1.f, 1.f), "specular color");
  addF("specular_roughness", 0.2f, 0.f, 1.f, "specular roughness");
  addF("specular_IOR", 1.5f, 1.f, 3.f, "specular index of refraction");
  addF("transmission", 0.f, 0.f, 1.f, "transmission weight");
  addF("emission", 0.f, 0.f, 1.f, "emission weight");
  addColor("emission_color", float3(1.f, 1.f, 1.f), "emission color");
  addF("coat", 0.f, 0.f, 1.f, "coat weight");
  addColor("opacity", float3(1.f, 1.f, 1.f),
      "opacity (consumed monochromatically by standard_surface)");
}

namespace tokens::material {

Token const matte = "matte";
Token const physicallyBased = "physicallyBased";
Token const physicallyBasedMDL = "physicallyBasedMDL";
Token const mdl = "mdl";
Token const materialx = "materialx";

} // namespace tokens::material

} // namespace tsd::scene
