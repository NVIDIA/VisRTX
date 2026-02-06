// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// std
#include <algorithm>
#include <cstring>
// stb
#include "stb_image.h"
#include "stb_image_write.h"
#ifndef _WIN32
#include "tinyexr.h"
#endif
// tsd
#include "tsd/core/Logging.hpp"

#include "HDRImage.h"

namespace tsd::io {

using namespace tsd::core;

bool HDRImage::import(std::string fileName)
{
  if (fileName.size() < 4)
    return false;

  // check the extension
  std::string extension = std::string(strrchr(fileName.c_str(), '.'));
  std::transform(extension.data(),
      extension.data() + extension.size(),
      std::addressof(extension[0]),
      [](unsigned char c) { return std::tolower(c); });

  if (extension != ".hdr" && extension != ".exr")
    return false;

  if (extension == ".hdr") {
    int w, h, n;
    stbi_set_flip_vertically_on_load(1);
    const float *imgData = stbi_loadf(fileName.c_str(), &w, &h, &n, STBI_rgb);
    stbi_set_flip_vertically_on_load(0); // Restore default top-down orientation
    width = w;
    height = h;
    numComponents = 3; // because of STBI_rgb
    if (width <= 0 || height <= 0 || n < 3) {
      stbi_image_free(const_cast<float*>(imgData));
      logError("import_HDRI] error importing HDR image: %s", fileName.c_str());
      return false;
    }

    pixel.resize(w * h * 3);
    std::memcpy(pixel.data(), imgData, w * h * 3 * sizeof(float));
    stbi_image_free(const_cast<float*>(imgData));
    return true;
#ifdef _WIN32
  }
#else
  } else {
    int w, h;
    float *imgData;
    const char *err;
    int ret = LoadEXR(&imgData, &w, &h, fileName.c_str(), &err);
    if (ret != 0) {
      logError("import_HDRI] error importing EXR: %s", err);
      return false;
    }

    width = w;
    height = h;
    numComponents = 3;

    pixel.resize(w * h * 3);
    // LoadEXR returns a RGBA image, we want RGB.
    // Vertical flip the image.
    for (auto j = 0; j < h; ++j) {
      for (auto i = 0; i < w; ++i) {
        auto srcidx = 4 * (j * w + i);
        auto dstidx = 3 * ((h - j - 1) * w + i);
        pixel[dstidx + 0] = imgData[srcidx + 0];
        pixel[dstidx + 1] = imgData[srcidx + 1];
        pixel[dstidx + 2] = imgData[srcidx + 2];
      }
    }
    free(imgData);

    return true;
  }
#endif

  return false;
}

} // namespace tsd
