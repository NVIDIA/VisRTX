// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/HDRImage.h"
#include "tsd/io/importers/detail/importer_common.hpp"

namespace tsd::io {

using namespace tsd::core;

void import_HDRI(Scene &scene, const char *filepath, LayerNodeRef location)
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

    auto arr = scene.createArray(ANARI_FLOAT32_VEC3, img.width, img.height);
    arr->setData(rgb.data());

    auto [inst, hdri] = scene.insertNewChildObjectNode<Light>(
        location ? location : scene.defaultLayer()->root(), tokens::light::hdri);
    hdri->setName(fileOf(filepath).c_str());
    hdri->setParameterObject("radiance", *arr);
  }
}

} // namespace tsd::io
