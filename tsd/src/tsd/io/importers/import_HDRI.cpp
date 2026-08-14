// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/HDRImage.h"
#include "tsd/io/importers/detail/importer_common.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"

namespace tsd::io {

using namespace tsd::core;

void import_HDRI(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filepath,
    LayerNodeRef location)
{
  (void)animMgr;
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

    ImageCache cache(&scene);
    // Stored bottom-up: an hdri light's radiance is mapped over the sphere by
    // the light rather than addressed by an image sampler, so the top-left
    // origin samplers are stored for does not apply to it.
    auto image = cache.acquireDecoded(
        {hdriFilename, ColorSpace::LINEAR, RowOrder::BOTTOM_UP},
        ANARI_FLOAT32_VEC3,
        img.width,
        img.height,
        img.rowOrder,
        rgb.data());

    if (!image) {
      logError("[import_HDRI] failed to store radiance for '%s'", filepath);
      return;
    }

    auto [inst, hdri] = scene.insertNewChildObjectNode<Light>(
        location ? location : scene.defaultLayer()->root(),
        tokens::light::hdri);
    hdri->setName(fileOf(filepath).c_str());
    hdri->setParameterObject("radiance", *image.texels);
  } else {
    tsd::core::logError("[import_HDRI] Failed to load file '%s'", filepath);
  }
}

} // namespace tsd::io
