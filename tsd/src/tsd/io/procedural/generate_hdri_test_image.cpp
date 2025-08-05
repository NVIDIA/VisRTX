// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"

namespace tsd::io {

void generate_hdri_test_image(Context &ctx, LayerNodeRef location)
{
  if (!location)
    location = ctx.defaultLayer()->root();

  int2 size(1024, 1024);

  auto arr = ctx.createArray(ANARI_FLOAT32_VEC3, size.x, size.y);

  auto *as3f = arr->mapAs<float3>();

  // ==================================================================
  // base pattern is thin black grid on white background
  // ==================================================================
  for (int iy = 0; iy < size.y; iy++) {
    for (int ix = 0; ix < size.x; ix++) {
      float3 color =
          ((ix % 16 == 0) || (iy % 16 == 0)) ? float3(0.f) : float3(1.f);
      as3f[ix + iy * size.x] = color;
    }
  }
  // ==================================================================
  // red square in lower left corner
  // ==================================================================
  for (int iy = 0; iy < size.y / 32; iy++) {
    for (int ix = 0; ix < size.x / 32; ix++) {
      float3 color = float3(1, 0, 0);
      as3f[ix + iy * size.x] = color;
    }
  }
  // ==================================================================
  // blue crosshair through center image,
  // ==================================================================
  for (int iy = 0; iy < size.y; iy++) {
    int ix = size.x / 2;
    as3f[ix + iy * size.x] = float3(0, 0, 1);
  }
  for (int ix = 0; ix < size.x; ix++) {
    int iy = size.y / 2;
    as3f[ix + iy * size.x] = float3(0, 0, 1);
  }
  // ==================================================================
  // gradient
  // ==================================================================
  int iy0 = size.y / 2 - 16;
  int ix0 = size.x / 2 - 16;
  int iy1 = size.y / 2 + 16;
  int ix1 = size.x / 2 + 16;
  for (int iy = iy0; iy <= iy1; iy++) {
    for (int ix = ix0; ix <= ix1; ix++) {
      float r = float(ix - ix0) / float(ix1 - ix0);
      float g = float(iy - iy0) / float(iy1 - iy0);
      // as3f[ix+iy*size.x] = float3(0,1,0);
      as3f[ix + iy * size.x] = float3(r, g, (r + g) / 2.f);
    }
  }

  arr->unmap();

  auto [inst, hdri] = ctx.insertNewChildObjectNode<Light>(
      location, tokens::light::hdri);
  hdri->setName("hdri_dome");
  hdri->setParameterObject("radiance"_t, *arr);
}

} // namespace tsd
