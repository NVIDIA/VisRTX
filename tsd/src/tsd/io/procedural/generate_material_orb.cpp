// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"
// data
#include "tsd/io/procedural/embedded_obj/TestOrb_base.h"
#include "tsd/io/procedural/embedded_obj/TestOrb_equation.h"
#include "tsd/io/procedural/embedded_obj/TestOrb_floor.h"
#include "tsd/io/procedural/embedded_obj/TestOrb_inner_sphere.h"
#include "tsd/io/procedural/embedded_obj/TestOrb_outer_sphere.h"

namespace tsd::io {

#define addObject(name, source, mat_type, mat)                                 \
  {                                                                            \
    auto mesh = scene.createObject<Geometry>(tokens::geometry::triangle);      \
    mesh->setName((std::string(name) + "_geometry").c_str());                  \
                                                                               \
    auto positionArray = scene.createArray(                                    \
        ANARI_FLOAT32_VEC3, std::size(source::vertex_position) / 3);           \
    positionArray->setData(std::data(source::vertex_position));                \
                                                                               \
    auto normalArray = scene.createArray(                                      \
        ANARI_FLOAT32_VEC3, std::size(source::vertex_normal) / 3);             \
    normalArray->setData(std::data(source::vertex_normal));                    \
                                                                               \
    auto uvArray = scene.createArray(                                          \
        ANARI_FLOAT32_VEC2, std::size(source::vertex_uv) / 2);                 \
    uvArray->setData(std::data(source::vertex_uv));                            \
                                                                               \
    mesh->setParameterObject("vertex.position", *positionArray);             \
    mesh->setParameterObject("vertex.normal", *normalArray);                 \
    mesh->setParameterObject("vertex.attribute0", *uvArray);                 \
                                                                               \
    mat = scene.createObject<Material>(mat_type);                              \
    auto matName = std::string(name) + "_material";                            \
    mat->setName(matName.c_str());                                             \
                                                                               \
    auto surface = scene.createSurface(name, mesh, mat);                       \
    scene.insertChildObjectNode(orb_root, surface);                            \
  }

static SamplerRef makeCheckboardTexture(Scene &scene, int size)
{
  auto tex = scene.createObject<Sampler>(tokens::sampler::image2D);

  auto array = scene.createArray(ANARI_FLOAT32_VEC4, size, size);
  auto *data = array->mapAs<tsd::math::float4>();

  constexpr auto lightGray = tsd::math::float4(.7f, .7f, .7f, 1.f);
  constexpr auto darkGray = tsd::math::float4(.3f, .3f, .3f, 1.f);
  for (int h = 0; h < size; h++) {
    for (int w = 0; w < size; w++) {
      bool even = h & 1;
      if (even)
        data[h * size + w] = w & 1 ? lightGray : darkGray;
      else
        data[h * size + w] = w & 1 ? darkGray : lightGray;
    }
  }
  array->unmap();

  tex->setParameterObject("image", *array);
  tex->setParameter("inAttribute", "attribute0");
  tex->setParameter("wrapMode1", "clampToEdge");
  tex->setParameter("wrapMode2", "clampToEdge");
  tex->setParameter("filter", "nearest");
  tex->setName("checkerboard");

  return tex;
}

///////////////////////////////////////////////////////////////////////////////

void generate_material_orb(Scene &scene, LayerNodeRef location)
{
  if (!location)
    location = scene.defaultLayer()->root();

  auto orb_root = location->insert_last_child(
      {tsd::math::mat4(tsd::math::identity)});
  (*orb_root)->name() = "Material Orb";

  MaterialRef mat;

  addObject(
      "base", obj2header::TestOrb_base, tokens::material::physicallyBased, mat);
  mat->setParameter("baseColor", tsd::math::float3(0.292f));
  mat->setParameter("metallic", 0.f);
  mat->setParameter("roughness", 0.f);
  mat->setParameter("clearcoat", 1.f);

  addObject("equation",
      obj2header::TestOrb_equation,
      tokens::material::physicallyBased,
      mat);
  mat->setParameter("baseColor", tsd::math::float3(0.775f, 0.759f, 0.f));
  mat->setParameter("metallic", 0.5f);
  mat->setParameter("roughness", 0.f);
  mat->setParameter("clearcoat", 1.f);

  addObject("inner_sphere",
      obj2header::TestOrb_inner_sphere,
      tokens::material::physicallyBased,
      mat);
  mat->setParameter("baseColor", tsd::math::float3(0.1f));
  mat->setParameter("metallic", 0.5f);
  mat->setParameter("roughness", 0.f);
  mat->setParameter("clearcoat", 1.f);

  addObject("outer_sphere",
      obj2header::TestOrb_outer_sphere,
      tokens::material::physicallyBased,
      mat);
  mat->setParameter("baseColor", tsd::math::float3(0.f, 0.110f, 0.321f));
  mat->setParameter("metallic", 0.5f);
  mat->setParameter("roughness", 0.f);
  mat->setParameter("clearcoat", 1.f);

  addObject("floor", obj2header::TestOrb_floor, tokens::material::matte, mat);
  auto tex = makeCheckboardTexture(scene, 10);
  mat->setParameterObject("color", *tex);
}

} // namespace tsd::io
