// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/procedural.hpp"
// data
#include "tsd/authoring/procedural/embedded_obj/TestOrb_base.h"
#include "tsd/authoring/procedural/embedded_obj/TestOrb_equation.h"
#include "tsd/authoring/procedural/embedded_obj/TestOrb_floor.h"
#include "tsd/authoring/procedural/embedded_obj/TestOrb_inner_sphere.h"
#include "tsd/authoring/procedural/embedded_obj/TestOrb_outer_sphere.h"

namespace tsd {

#define addObject(name, source, mat)                                           \
  {                                                                            \
    auto mesh = ctx.createObject<Geometry>(tokens::geometry::triangle);        \
    mesh->setName((std::string(name) + "_geometry").c_str());                  \
                                                                               \
    auto positionArray = ctx.createArray(                                      \
        ANARI_FLOAT32_VEC3, source::vertex_position.size() / 3);               \
    positionArray->setData(source::vertex_position.data());                    \
                                                                               \
    auto normalArray =                                                         \
        ctx.createArray(ANARI_FLOAT32_VEC3, source::vertex_normal.size() / 3); \
    normalArray->setData(source::vertex_normal.data());                        \
                                                                               \
    auto uvArray =                                                             \
        ctx.createArray(ANARI_FLOAT32_VEC2, source::vertex_uv.size() / 2);     \
    uvArray->setData(source::vertex_uv.data());                                \
                                                                               \
    mesh->setParameterObject("vertex.position"_t, *positionArray);             \
    mesh->setParameterObject("vertex.normal"_t, *normalArray);                 \
    mesh->setParameterObject("vertex.attribute0"_t, *uvArray);                 \
                                                                               \
    mat = ctx.createObject<Material>(tokens::material::physicallyBased);       \
    auto matName = std::string(name) + "_material";                            \
    mat->setName(matName.c_str());                                             \
    mat->setParameter("baseColor"_t, float3(1.f));                             \
                                                                               \
    auto surface = ctx.createSurface(name, mesh, mat);                         \
                                                                               \
    ctx.tree.insert_last_child(                                                \
        orb_root, utility::Any(ANARI_SURFACE, surface.index()));               \
  }

static IndexedVectorRef<Sampler> makeCheckboardTexture(Context &ctx, int size)
{
  auto tex = ctx.createObject<Sampler>(tokens::sampler::image2D);

  auto array = ctx.createArray(ANARI_FLOAT32_VEC3, size, size);
  auto *data = array->mapAs<tsd::float3>();
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

void generate_material_orb(Context &ctx)
{
  auto orb_root = ctx.tree.insert_last_child(
      ctx.tree.root(), {tsd::mat4(tsd::math::identity), "material_orb"});

  IndexedVectorRef<Material> mat;

  addObject("base", obj2header::TestOrb_base, mat);
  mat->setParameter("baseColor"_t, tsd::math::float3(0.292f));
  mat->setParameter("metallic"_t, 0.f);
  mat->setParameter("roughness"_t, 0.f);
  mat->setParameter("clearcoat"_t, 1.f);

  addObject("equation", obj2header::TestOrb_equation, mat);
  mat->setParameter("baseColor"_t, tsd::math::float3(0.775f, 0.759f, 0.f));
  mat->setParameter("metallic"_t, 0.5f);
  mat->setParameter("roughness"_t, 0.f);
  mat->setParameter("clearcoat"_t, 1.f);

  addObject("inner_sphere", obj2header::TestOrb_inner_sphere, mat);
  mat->setParameter("baseColor"_t, tsd::math::float3(0.1f));
  mat->setParameter("metallic"_t, 0.5f);
  mat->setParameter("roughness"_t, 0.f);
  mat->setParameter("clearcoat"_t, 1.f);

  addObject("outer_sphere", obj2header::TestOrb_outer_sphere, mat);
  mat->setParameter("baseColor"_t, tsd::math::float3(0.f, 0.110f, 0.321f));
  mat->setParameter("metallic"_t, 0.5f);
  mat->setParameter("roughness"_t, 0.f);
  mat->setParameter("clearcoat"_t, 1.f);

  addObject("floor", obj2header::TestOrb_floor, mat);
  auto tex = makeCheckboardTexture(ctx, 10);
  mat->setParameterObject("baseColor"_t, *tex);
  mat->setParameter("metallic"_t, 0.f);
  mat->setParameter("roughness"_t, 1.f);
  mat->setParameter("clearcoat"_t, 0.f);
}

} // namespace tsd
