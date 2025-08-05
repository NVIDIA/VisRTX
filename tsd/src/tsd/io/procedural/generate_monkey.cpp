// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"
#include "tsd/io/procedural/embedded_obj/monkey.h"

namespace tsd::io {

void generate_monkey(Context &ctx, LayerNodeRef location)
{
  if (!location)
    location = ctx.defaultLayer()->root();

  auto monkey = ctx.createObject<Geometry>(tokens::geometry::triangle);
  monkey->setName("monkey_geometry");

  auto positionArray = ctx.createArray(
      ANARI_FLOAT32_VEC3, std::size(obj2header::vertex_position) / 3);
  positionArray->setData(std::data(obj2header::vertex_position));

  auto normalArray = ctx.createArray(
      ANARI_FLOAT32_VEC3, std::size(obj2header::vertex_normal) / 3);
  normalArray->setData(std::data(obj2header::vertex_normal));

  auto uvArray =
      ctx.createArray(ANARI_FLOAT32_VEC2, std::size(obj2header::vertex_uv) / 2);
  uvArray->setData(std::data(obj2header::vertex_uv));

  auto indexArray = ctx.createArray(
      ANARI_UINT32_VEC3, std::size(obj2header::primitive_index) / 3);
  indexArray->setData(std::data(obj2header::primitive_index));

  monkey->setParameterObject("vertex.position"_t, *positionArray);
#if 0 // NOTE: these appear to be wrong
  monkey->setParameterObject("vertex.normal"_t, *normalArray);
#endif
  monkey->setParameterObject("vertex.attribute0"_t, *uvArray);
  monkey->setParameterObject("primitive.index"_t, *indexArray);

  auto surface = ctx.createSurface("monkey", monkey);
  ctx.insertChildObjectNode(location, surface);
}

} // namespace tsd
