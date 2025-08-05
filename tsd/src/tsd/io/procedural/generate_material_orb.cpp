// Copyright 2024-2025 NVIDIA Corporation
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
    auto mesh = ctx.createObject<Geometry>(tokens::geometry::triangle);        \
    mesh->setName((std::string(name) + "_geometry").c_str());                  \
                                                                               \
    auto positionArray = ctx.createArray(                                      \
        ANARI_FLOAT32_VEC3, std::size(source::vertex_position) / 3);           \
    positionArray->setData(std::data(source::vertex_position));                \
                                                                               \
    auto normalArray = ctx.createArray(                                        \
        ANARI_FLOAT32_VEC3, std::size(source::vertex_normal) / 3);             \
    normalArray->setData(std::data(source::vertex_normal));                    \
                                                                               \
    auto uvArray =                                                             \
        ctx.createArray(ANARI_FLOAT32_VEC2, std::size(source::vertex_uv) / 2); \
    uvArray->setData(std::data(source::vertex_uv));                            \
                                                                               \
    mesh->setParameterObject("vertex.position"_t, *positionArray);             \
    mesh->setParameterObject("vertex.normal"_t, *normalArray);                 \
    mesh->setParameterObject("vertex.attribute0"_t, *uvArray);                 \
                                                                               \
    mat = ctx.createObject<Material>(mat_type);                                \
    auto matName = std::string(name) + "_material";                            \
    mat->setName(matName.c_str());                                             \
                                                                               \
    auto surface = ctx.createSurface(name, mesh, mat);                         \
    ctx.insertChildObjectNode(orb_root, surface);                              \
  }

static SamplerRef makeCheckboardTexture(Context &ctx, int size)
{
  auto tex = ctx.createObject<Sampler>(tokens::sampler::image2D);

  auto array = ctx.createArray(ANARI_FLOAT32_VEC3, size, size);
  auto *data = array->mapAs<tsd::math::float3>();
  for (int h = 0; h < size; h++) {
    for (int w = 0; w < size; w++) {
      bool even = h & 1;
      if (even)
        data[h * size + w] =
            w & 1 ? tsd::math::float3(.7f) : tsd::math::float3(.3f);
      else
        data[h * size + w] =
            w & 1 ? tsd::math::float3(.3f) : tsd::math::float3(.7f);
    }
  }
  array->unmap();

  tex->setParameterObject("image"_t, *array);
  tex->setParameter("inAttribute"_t, "attribute0");
  tex->setParameter("wrapMode1"_t, "clampToEdge");
  tex->setParameter("wrapMode2"_t, "clampToEdge");
  tex->setParameter("filter"_t, "nearest");
  tex->setName("checkerboard");

  return tex;
}

///////////////////////////////////////////////////////////////////////////////

void generate_material_orb(Context &ctx, LayerNodeRef location)
{
  if (!location)
    location = ctx.defaultLayer()->root();

  auto orb_root = location->insert_last_child(
      {tsd::math::mat4(tsd::math::identity), "material_orb"});

  MaterialRef mat;

  addObject(
      "base", obj2header::TestOrb_base, tokens::material::physicallyBased, mat);
  mat->setParameter("baseColor"_t, tsd::math::float3(0.292f));
  mat->setParameter("metallic"_t, 0.f);
  mat->setParameter("roughness"_t, 0.f);
  mat->setParameter("clearcoat"_t, 1.f);

  addObject("equation",
      obj2header::TestOrb_equation,
      tokens::material::physicallyBased,
      mat);
  mat->setParameter("baseColor"_t, tsd::math::float3(0.775f, 0.759f, 0.f));
  mat->setParameter("metallic"_t, 0.5f);
  mat->setParameter("roughness"_t, 0.f);
  mat->setParameter("clearcoat"_t, 1.f);

  addObject("inner_sphere",
      obj2header::TestOrb_inner_sphere,
      tokens::material::physicallyBased,
      mat);
  mat->setParameter("baseColor"_t, tsd::math::float3(0.1f));
  mat->setParameter("metallic"_t, 0.5f);
  mat->setParameter("roughness"_t, 0.f);
  mat->setParameter("clearcoat"_t, 1.f);

  addObject("outer_sphere",
      obj2header::TestOrb_outer_sphere,
      tokens::material::physicallyBased,
      mat);
  mat->setParameter("baseColor"_t, tsd::math::float3(0.f, 0.110f, 0.321f));
  mat->setParameter("metallic"_t, 0.5f);
  mat->setParameter("roughness"_t, 0.f);
  mat->setParameter("clearcoat"_t, 1.f);

  addObject("floor", obj2header::TestOrb_floor, tokens::material::matte, mat);
  auto tex = makeCheckboardTexture(ctx, 10);
  mat->setParameterObject("color"_t, *tex);
}

} // namespace tsd
