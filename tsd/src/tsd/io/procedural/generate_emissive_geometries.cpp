// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// A dark room lit ONLY by emissive geometries, so each emitter's cast light and
// color bleed are directly visible. Emitters sit in a row above a neutral floor
// with a back wall behind them; every emitter type drops a colored pool on the
// floor and tints the wall:
//   - constant warm quad          (uniform float3 emission)
//   - constant cool sphere        (analytic sphere area light)
//   - RGB checker quad            (textured emission, sharp)
//   - horizontal gradient quad    (textured emission, smooth)
//   - ANARI quad light            (light-primitive reference, same footprint)
//
// Pair with generate_default_lights disabled / a dark renderer background to see
// the effect. See TestEmissiveGeometryLight.cpp for the quantitative
// counterpart.

#include "tsd/io/procedural.hpp"
// std
#include <array>

namespace tsd::io {

namespace {

constexpr float kEmitterY = 0.9f; // emitters float this high above the floor
constexpr float kEmitterHalf = 0.4f;
constexpr int kTexRes = 16;

using Quad = std::array<float3, 4>;

// Horizontal (XZ-plane) quad centered at (cx, cy, cz), facing -Y (down) so its
// light falls on the floor below.
Quad horizontalQuad(float cx, float cy, float cz, float hx, float hz)
{
  return {float3(cx - hx, cy, cz - hz),
      float3(cx + hx, cy, cz - hz),
      float3(cx + hx, cy, cz + hz),
      float3(cx - hx, cy, cz + hz)};
}

SurfaceRef addQuad(Scene &scene,
    LayerNodeRef root,
    const char *name,
    const Quad &pos,
    const float3 &normal,
    MaterialRef mat,
    bool withUV)
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

  if (withUV) {
    std::array<float2, 4> uv = {
        float2(0.f, 0.f), float2(1.f, 0.f), float2(1.f, 1.f), float2(0.f, 1.f)};
    auto uvArray = scene.createArray(ANARI_FLOAT32_VEC2, uv.size());
    uvArray->setData(uv.data());
    geom->setParameterObject("vertex.attribute0", *uvArray);
  }

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

MaterialRef makeConstantEmitter(
    Scene &scene, const char *name, const float3 &emissive)
{
  auto mat = scene.createObject<Material>(tokens::material::physicallyBased);
  mat->setName(name);
  mat->setParameter("baseColor", float3(0.f));
  mat->setParameter("metallic", 0.f);
  mat->setParameter("roughness", 1.f);
  mat->setParameter("emissive", emissive);
  return mat;
}

// image2D sampler filled by `fill(u, v) -> emitted color`, mapped via
// attribute0. `nearest` for the checker, `linear` for the gradient.
template <typename FillFn>
SamplerRef makeEmissionTexture(
    Scene &scene, const char *name, const char *filter, FillFn fill)
{
  auto tex = scene.createObject<Sampler>(tokens::sampler::image2D);
  tex->setName(name);

  auto array = scene.createArray(ANARI_FLOAT32_VEC4, kTexRes, kTexRes);
  auto *data = array->mapAs<float4>();
  for (int h = 0; h < kTexRes; ++h) {
    for (int w = 0; w < kTexRes; ++w) {
      const float u = (w + 0.5f) / kTexRes;
      const float v = (h + 0.5f) / kTexRes;
      const float3 c = fill(u, v);
      data[h * kTexRes + w] = float4(c.x, c.y, c.z, 1.f);
    }
  }
  array->unmap();

  tex->setParameterObject("image", *array);
  tex->setParameter("inAttribute", "attribute0");
  tex->setParameter("wrapMode1", "clampToEdge");
  tex->setParameter("wrapMode2", "clampToEdge");
  tex->setParameter("filter", filter);
  return tex;
}

MaterialRef makeTexturedEmitter(
    Scene &scene, const char *name, SamplerRef emissionTex)
{
  auto mat = scene.createObject<Material>(tokens::material::physicallyBased);
  mat->setName(name);
  mat->setParameter("baseColor", float3(0.f));
  mat->setParameter("metallic", 0.f);
  mat->setParameter("roughness", 1.f);
  mat->setParameterObject("emissive", *emissionTex);
  return mat;
}

} // namespace

void generate_emissive_geometries(Scene &scene, LayerNodeRef location)
{
  if (!location)
    location = scene.defaultLayer()->root();
  auto *layer = (*location)->layer();

  auto root = location->insert_last_child(
      {layer, math::IDENTITY_MAT4, "Emissive Geometries"});

  // Receivers: a neutral floor (+Y up) and a back wall (+Z toward the row) so
  // both the cast pools and vertical color bleed are visible.
  addQuad(scene,
      root,
      "floor",
      {float3(-5.f, 0.f, -4.f),
          float3(5.f, 0.f, -4.f),
          float3(5.f, 0.f, 3.f),
          float3(-5.f, 0.f, 3.f)},
      float3(0.f, 1.f, 0.f),
      makeDiffuse(scene, "floor_material", float3(0.6f)),
      false);
  addQuad(scene,
      root,
      "backWall",
      {float3(-5.f, 0.f, -3.f),
          float3(5.f, 0.f, -3.f),
          float3(5.f, 3.f, -3.f),
          float3(-5.f, 3.f, -3.f)},
      float3(0.f, 0.f, 1.f),
      makeDiffuse(scene, "backWall_material", float3(0.6f)),
      false);

  const float3 down(0.f, -1.f, 0.f);

  // Slot 0 — constant warm quad emitter.
  addQuad(scene,
      root,
      "warmQuad",
      horizontalQuad(-3.f, kEmitterY, 0.f, kEmitterHalf, kEmitterHalf),
      down,
      makeConstantEmitter(scene, "warmQuad_material", float3(12.f, 4.5f, 1.2f)),
      false);

  // Slot 1 — constant cool sphere emitter (analytic sphere area light).
  {
    auto geom = scene.createObject<Geometry>(tokens::geometry::sphere);
    geom->setName("coolSphere_geometry");
    auto center = scene.createArray(ANARI_FLOAT32_VEC3, 1);
    float3 c(-1.5f, kEmitterY, 0.f);
    center->setData(&c);
    geom->setParameterObject("vertex.position", *center);
    geom->setParameter("radius", 0.3f);
    auto surface = scene.createSurface("coolSphere",
        geom,
        makeConstantEmitter(scene, "coolSphere_material", float3(0.9f, 3.f, 12.f)));
    scene.insertChildObjectNode(root, surface);
  }

  // Slot 2 — RGB checker (sharp textured emission).
  {
    auto tex = makeEmissionTexture(
        scene, "checker_emission", "nearest", [](float u, float v) {
          const bool even = (int(u * 4) + int(v * 4)) & 1;
          return even ? float3(9.f, 0.3f, 9.f) : float3(0.3f, 9.f, 0.9f);
        });
    addQuad(scene,
        root,
        "checkerQuad",
        horizontalQuad(0.f, kEmitterY, 0.f, kEmitterHalf, kEmitterHalf),
        down,
        makeTexturedEmitter(scene, "checkerQuad_material", tex),
        true);
  }

  // Slot 3 — horizontal red->blue gradient (smooth textured emission).
  {
    auto tex = makeEmissionTexture(
        scene, "gradient_emission", "linear", [](float u, float) {
          return float3(9.f * (1.f - u), 0.5f, 9.f * u);
        });
    addQuad(scene,
        root,
        "gradientQuad",
        horizontalQuad(1.5f, kEmitterY, 0.f, kEmitterHalf, kEmitterHalf),
        down,
        makeTexturedEmitter(scene, "gradientQuad_material", tex),
        true);
  }

  // Slot 4 — ANARI quad light reference: same footprint/height as the emitters,
  // emitting downward, for side-by-side comparison with the emissive geometries.
  {
    auto light = scene.createObject<Light>(tokens::light::quad);
    light->setName("referenceQuadLight");
    light->setParameter("color", float3(1.f, 0.9f, 0.7f));
    light->setParameter("position",
        float3(3.f - kEmitterHalf, kEmitterY, -kEmitterHalf));
    light->setParameter("edge1", float3(2.f * kEmitterHalf, 0.f, 0.f));
    light->setParameter("edge2", float3(0.f, 0.f, 2.f * kEmitterHalf));
    light->setParameter("intensity", 9.f);
    light->setParameter("side", "both");
    scene.insertChildObjectNode(root, light);
  }
}

} // namespace tsd::io
