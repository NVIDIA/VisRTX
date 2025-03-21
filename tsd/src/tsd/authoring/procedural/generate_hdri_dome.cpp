// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/procedural.hpp"

namespace tsd {

void generate_hdri_dome(Context &ctx, LayerNodeRef location)
{
  if (!location)
    location = ctx.defaultLayer()->root();

  auto arr = ctx.createArray(ANARI_FLOAT32_VEC3, 1, 2);
  std::vector<float3> colors = {float3(0.1f), float3(0.8f, 0.8f, 0.8f)};
  arr->setData(colors.data());

  auto [inst, hdri] = ctx.insertNewChildObjectNode<tsd::Light>(
      location, tsd::tokens::light::hdri);
  hdri->setName("hdri_dome");
  hdri->setParameterObject("radiance"_t, *arr);
}

} // namespace tsd
