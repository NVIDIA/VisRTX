// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/procedural.hpp"
#include "tsd/authoring/procedural/embedded_obj/monkey.h"

namespace tsd {

void generate_monkey(Context &ctx, InstanceNode::Ref location)
{
  if (!location)
    location = ctx.tree.root();

  auto monkey = ctx.createObject<Geometry>(tokens::geometry::triangle);
  monkey->setName("monkey_geometry");

  auto positionArray = ctx.createArray(
      ANARI_FLOAT32_VEC3, obj2header::vertex_position.size() / 3);
  positionArray->setData(obj2header::vertex_position.data());

  auto normalArray =
      ctx.createArray(ANARI_FLOAT32_VEC3, obj2header::vertex_normal.size() / 3);
  normalArray->setData(obj2header::vertex_normal.data());

  auto uvArray =
      ctx.createArray(ANARI_FLOAT32_VEC2, obj2header::vertex_uv.size() / 2);
  uvArray->setData(obj2header::vertex_uv.data());

  auto indexArray = ctx.createArray(
      ANARI_UINT32_VEC3, obj2header::primitive_index.size() / 3);
  indexArray->setData(obj2header::primitive_index.data());

  monkey->setParameterObject("vertex.position"_t, *positionArray);
#if 0 // NOTE: these appear to be wrong
  monkey->setParameterObject("vertex.normal"_t, *normalArray);
#endif
  monkey->setParameterObject("vertex.attribute0"_t, *uvArray);
  monkey->setParameterObject("primitive.index"_t, *indexArray);

  auto surface = ctx.createSurface("monkey", monkey);

  ctx.tree.insert_last_child(
      location, utility::Any(ANARI_SURFACE, surface.index()));
}

} // namespace tsd
