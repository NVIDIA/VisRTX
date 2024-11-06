// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"

#include "tsd/authoring/importers/detail/HDRImage.h"

namespace tsd {

void import_HDRI(Context &ctx, const char *filepath)
{
  std::string hdriFilename = filepath;
  HDRImage img;
  if (img.import(hdriFilename)) {
    std::vector<float3> rgb(img.width * img.height);

    if (img.numComponents == 3) {
      memcpy(rgb.data(), img.pixel.data(), sizeof(rgb[0]) * rgb.size());
    } else if (img.numComponents == 4) {
      for (size_t i = 0; i < img.pixel.size(); i += 4) {
        rgb[i / 4] = float3(img.pixel[i], img.pixel[i + 1], img.pixel[i + 2]);
      }
    }

    tsd::IndexedVectorRef<tsd::Array> arr =
        ctx.createArray(ANARI_FLOAT32_VEC3, img.width, img.height);
    arr->setData(rgb.data());

    auto hdri = ctx.createObject<tsd::Light>("hdri"_t);
    hdri->setName("HDRI");
    hdri->setParameterObject("radiance"_t, *arr);
    ctx.tree.insert_last_child(
        ctx.tree.root(), tsd::utility::Any(ANARI_LIGHT, hdri.index()));
  }
}

} // namespace tsd
