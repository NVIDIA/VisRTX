// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"

namespace tsd::io {

void generate_hdri_dome(Scene &scene, LayerNodeRef location)
{
  if (!location)
    location = scene.defaultLayer()->root();

  auto arr = scene.createArray(ANARI_FLOAT32_VEC3, 1, 2);
  std::vector<float3> colors = {float3(0.1f), float3(0.8f, 0.8f, 0.8f)};
  arr->setData(colors.data());

  auto [inst, hdri] = scene.insertNewChildObjectNode<Light>(
      location, tokens::light::hdri);
  hdri->setName("hdri_dome");
  hdri->setParameterObject("radiance", *arr);
}

} // namespace tsd
