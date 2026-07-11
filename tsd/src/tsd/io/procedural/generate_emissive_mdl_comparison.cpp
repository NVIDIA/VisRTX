// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Side-by-side comparison of the three emissive-material implementations, so
// their Geometry Lights can be judged against each other visually (see
// TestEmissiveMaterialParity.cpp for the quantitative counterpart):
//
//              physicallyBased   physicallyBasedMDL   mdl (inline source)
//   constant        quad              quad                quad
//   sampled         quad              quad                quad
//
// Every emitter targets the SAME emitted radiance, so all six floor pools must
// match. The `mdl` column authors its intensity as `value * math::PI`: MDL
// emission intensity is radiant EXITANCE, and a diffuse EDF's radiance is
// intensity/PI — forgetting the PI factor makes an MDL emitter look ~3x dimmer
// than the equivalent physicallyBased one.
//
// All six emitters — the physicallyBasedMDL SAMPLED one included, its bound
// sampler being next-event sampled via the live sampler-mean Pick Power (ADR
// 0006) — must agree on both the 'quality' and 'interactive' renderers.

#include "tsd/io/procedural.hpp"
// std
#include <array>
#include <string>

namespace tsd::io {

namespace {

constexpr float kEmitterY = 0.9f;
constexpr float kEmitterHalf = 0.4f;
constexpr float kColumnPitch = 2.4f; // x spacing between implementations
constexpr float kRowPitch = 2.0f; // z spacing between emission kinds
constexpr int kTexRes = 16;
// Target emitted radiance; the checker's bright texel is 2x so its mean over
// the emitter stays at the same value as the constant row.
constexpr float kRadiance = 8.f;

using Quad = std::array<float3, 4>;

Quad horizontalQuad(float cx, float cy, float cz, float half)
{
  return {float3(cx - half, cy, cz - half),
      float3(cx + half, cy, cz - half),
      float3(cx + half, cy, cz + half),
      float3(cx - half, cy, cz + half)};
}

SurfaceRef addQuad(Scene &scene,
    LayerNodeRef root,
    const char *name,
    const Quad &pos,
    const float3 &normal,
    MaterialRef mat)
{
  auto geom = scene.createObject<Geometry>(tokens::geometry::triangle);
  geom->setName((std::string(name) + "_geometry").c_str());

  auto positionArray = scene.createArray(ANARI_FLOAT32_VEC3, pos.size());
  positionArray->setData(pos.data());
  geom->setParameterObject("vertex.position", *positionArray);

  std::array<float3, 4> normals = {normal, normal, normal, normal};
  auto normalArray = scene.createArray(ANARI_FLOAT32_VEC3, normals.size());
  normalArray->setData(normals.data());
  geom->setParameterObject("vertex.normal", *normalArray);

  std::array<float2, 4> uv = {
      float2(0.f, 0.f), float2(1.f, 0.f), float2(1.f, 1.f), float2(0.f, 1.f)};
  auto uvArray = scene.createArray(ANARI_FLOAT32_VEC2, uv.size());
  uvArray->setData(uv.data());
  geom->setParameterObject("vertex.attribute0", *uvArray);

  std::array<uint3, 2> idx = {uint3(0, 1, 2), uint3(0, 2, 3)};
  auto indexArray = scene.createArray(ANARI_UINT32_VEC3, idx.size());
  indexArray->setData(idx.data());
  geom->setParameterObject("primitive.index", *indexArray);

  auto surface = scene.createSurface(name, geom, mat);
  scene.insertChildObjectNode(root, surface);
  return surface;
}

MaterialRef makeDiffuse(Scene &scene, const char *name, const float3 &color)
{
  auto mat = scene.createObject<Material>(tokens::material::physicallyBased);
  mat->setName(name);
  mat->setParameter("baseColor", color);
  mat->setParameter("metallic", 0.f);
  mat->setParameter("roughness", 1.f);
  return mat;
}

// physicallyBased / physicallyBasedMDL constant emitter at `kRadiance`.
MaterialRef makeConstantEmitter(Scene &scene, const char *name, Token subtype)
{
  auto mat = scene.createObject<Material>(subtype);
  mat->setName(name);
  mat->setParameter("baseColor", float3(0.f));
  mat->setParameter("metallic", 0.f);
  mat->setParameter("roughness", 1.f);
  mat->setParameter("emissive", float3(kRadiance));
  return mat;
}

// The shared emission texture: a 2-level checker whose texels are 2*kRadiance
// and 0, so its mean equals the constant row. Nearest-filtered to keep the
// texel values (and therefore the mean) exact.
SamplerRef makeEmissionTexture(Scene &scene)
{
  auto tex = scene.createObject<Sampler>(tokens::sampler::image2D);
  tex->setName("mdlComparison_emission");

  auto array = scene.createArray(ANARI_FLOAT32_VEC4, kTexRes, kTexRes);
  auto *data = array->mapAs<float4>();
  for (int h = 0; h < kTexRes; ++h) {
    for (int w = 0; w < kTexRes; ++w) {
      const float v = ((w + h) & 1) ? 2.f * kRadiance : 0.f;
      data[h * kTexRes + w] = float4(v, v, v, 1.f);
    }
  }
  array->unmap();

  tex->setParameterObject("image", *array);
  tex->setParameter("inAttribute", "attribute0");
  tex->setParameter("wrapMode1", "clampToEdge");
  tex->setParameter("wrapMode2", "clampToEdge");
  tex->setParameter("filter", "nearest");
  return tex;
}

// physicallyBased / physicallyBasedMDL sampled emitter.
MaterialRef makeSampledEmitter(
    Scene &scene, const char *name, Token subtype, SamplerRef emissionTex)
{
  auto mat = scene.createObject<Material>(subtype);
  mat->setName(name);
  mat->setParameter("baseColor", float3(0.f));
  mat->setParameter("metallic", 0.f);
  mat->setParameter("roughness", 1.f);
  mat->setParameterObject("emissive", *emissionTex);
  return mat;
}

// Raw `mdl` constant emitter. Intensity is radiant exitance: author
// `value * PI` so the emitted radiance equals `value` — matching the
// physicallyBased `emissive` convention.
constexpr const char *kMdlConstantSource = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
export material emissive(color value = color(1.0)) = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: value * math::PI)));
)mdl";

MaterialRef makeMdlConstantEmitter(Scene &scene, const char *name)
{
  auto mat = scene.createObject<Material>(tokens::material::mdl);
  mat->setName(name);
  mat->setParameter("sourceType", "code");
  mat->setParameter("source", kMdlConstantSource);
  mat->setParameter("materialName", "emissive");
  mat->setParameter("value", float3(kRadiance));
  return mat;
}

// Raw `mdl` sampled emitter: same PI convention over a texture lookup driven
// by the first texture coordinate (attribute0 on the geometry).
constexpr const char *kMdlSampledSource = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
import ::state::*;
import ::tex::*;
export material emissive_tex(uniform texture_2d tex = texture_2d()) = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: tex::lookup_color(
                tex: tex,
                coord: float2(
                    state::texture_coordinate(0).x,
                    state::texture_coordinate(0).y)) * math::PI)));
)mdl";

MaterialRef makeMdlSampledEmitter(
    Scene &scene, const char *name, SamplerRef emissionTex)
{
  auto mat = scene.createObject<Material>(tokens::material::mdl);
  mat->setName(name);
  mat->setParameter("sourceType", "code");
  mat->setParameter("source", kMdlSampledSource);
  mat->setParameter("materialName", "emissive_tex");
  mat->setParameterObject("tex", *emissionTex);
  return mat;
}

} // namespace

void generate_emissive_mdl_comparison(Scene &scene, LayerNodeRef location)
{
  if (!location)
    location = scene.defaultLayer()->root();
  auto *layer = (*location)->layer();

  auto root = location->insert_last_child(
      {layer, math::IDENTITY_MAT4, "Emissive MDL Comparison"});

  // Receivers: neutral floor + back wall so the six pools sit side by side.
  addQuad(scene,
      root,
      "floor",
      {float3(-5.f, 0.f, -4.f),
          float3(5.f, 0.f, -4.f),
          float3(5.f, 0.f, 4.f),
          float3(-5.f, 0.f, 4.f)},
      float3(0.f, 1.f, 0.f),
      makeDiffuse(scene, "floor_material", float3(0.6f)));
  addQuad(scene,
      root,
      "backWall",
      {float3(-5.f, 0.f, -4.f),
          float3(5.f, 0.f, -4.f),
          float3(5.f, 3.f, -4.f),
          float3(-5.f, 3.f, -4.f)},
      float3(0.f, 0.f, 1.f),
      makeDiffuse(scene, "backWall_material", float3(0.6f)));

  const float3 down(0.f, -1.f, 0.f);
  const float xPBR = -kColumnPitch;
  const float xWrapper = 0.f;
  const float xMdl = kColumnPitch;
  const float zConstant = -0.5f * kRowPitch;
  const float zSampled = 0.5f * kRowPitch;

  // Row 1 — constant emission: all three implementations must match, on both
  // renderers.
  addQuad(scene,
      root,
      "constant_physicallyBased",
      horizontalQuad(xPBR, kEmitterY, zConstant, kEmitterHalf),
      down,
      makeConstantEmitter(scene,
          "constant_physicallyBased_material",
          tokens::material::physicallyBased));
  addQuad(scene,
      root,
      "constant_physicallyBasedMDL",
      horizontalQuad(xWrapper, kEmitterY, zConstant, kEmitterHalf),
      down,
      makeConstantEmitter(scene,
          "constant_physicallyBasedMDL_material",
          tokens::material::physicallyBasedMDL));
  addQuad(scene,
      root,
      "constant_mdl",
      horizontalQuad(xMdl, kEmitterY, zConstant, kEmitterHalf),
      down,
      makeMdlConstantEmitter(scene, "constant_mdl_material"));

  // Row 2 — sampled emission over one shared checker (mean == constant row).
  // All three implementations must match row 1 on both renderers.
  auto emissionTex = makeEmissionTexture(scene);
  addQuad(scene,
      root,
      "sampled_physicallyBased",
      horizontalQuad(xPBR, kEmitterY, zSampled, kEmitterHalf),
      down,
      makeSampledEmitter(scene,
          "sampled_physicallyBased_material",
          tokens::material::physicallyBased,
          emissionTex));
  addQuad(scene,
      root,
      "sampled_physicallyBasedMDL",
      horizontalQuad(xWrapper, kEmitterY, zSampled, kEmitterHalf),
      down,
      makeSampledEmitter(scene,
          "sampled_physicallyBasedMDL_material",
          tokens::material::physicallyBasedMDL,
          emissionTex));
  addQuad(scene,
      root,
      "sampled_mdl",
      horizontalQuad(xMdl, kEmitterY, zSampled, kEmitterHalf),
      down,
      makeMdlSampledEmitter(scene, "sampled_mdl_material", emissionTex));
}

} // namespace tsd::io
