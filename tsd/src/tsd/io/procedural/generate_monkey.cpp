// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"
#include "tsd/io/procedural/embedded_obj/monkey.h"

namespace tsd::io {

void generate_monkey(Scene &scene, LayerNodeRef location)
{
  if (!location)
    location = scene.defaultLayer()->root();

  auto monkey = scene.createObject<Geometry>(tokens::geometry::triangle);
  monkey->setName("monkey_geometry");

  auto positionArray = scene.createArray(
      ANARI_FLOAT32_VEC3, std::size(obj2header::vertex_position) / 3);
  positionArray->setData(std::data(obj2header::vertex_position));

  auto normalArray = scene.createArray(
      ANARI_FLOAT32_VEC3, std::size(obj2header::vertex_normal) / 3);
  normalArray->setData(std::data(obj2header::vertex_normal));

  auto uvArray =
      scene.createArray(ANARI_FLOAT32_VEC2, std::size(obj2header::vertex_uv) / 2);
  uvArray->setData(std::data(obj2header::vertex_uv));

  auto indexArray = scene.createArray(
      ANARI_UINT32_VEC3, std::size(obj2header::primitive_index) / 3);
  indexArray->setData(std::data(obj2header::primitive_index));

  monkey->setParameterObject("vertex.position", *positionArray);
#if 0 // NOTE: these appear to be wrong
  monkey->setParameterObject("vertex.normal", *normalArray);
#endif
  monkey->setParameterObject("vertex.attribute0", *uvArray);
  monkey->setParameterObject("primitive.index", *indexArray);

  auto surface = scene.createSurface("monkey", monkey);
  scene.insertChildObjectNode(location, surface);
}

} // namespace tsd
