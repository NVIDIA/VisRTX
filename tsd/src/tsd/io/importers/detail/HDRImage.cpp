// Copyright 2024-2025 NVIDIA Corporation
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
    stbi_set_flip_vertically_on_load(1);
    int w, h, n;
    float *imgData = stbi_loadf(fileName.c_str(), &w, &h, &n, STBI_rgb);
    width = w;
    height = h;
    numComponents = n;
    pixel.resize(w * h * n);
    std::memcpy(pixel.data(), imgData, w * h * n * sizeof(float));
    stbi_image_free(imgData);
    return width > 0 && height > 0
        && (numComponents == 3 || numComponents == 4);
#ifdef _WIN32
  }
#else
  } else {
    int w, h, n;
    float *imgData;
    const char *err;
    int ret = LoadEXR(&imgData, &w, &h, fileName.c_str(), &err);
    if (ret != 0) {
      logError("import_HDRI] error importing EXR: %s", err);
      return false;
    }
    n = 4;

    width = w;
    height = h;
    numComponents = n;
    pixel.resize(w * h * n);
    // flip-y
    const size_t rowStride = w * n;
    for (int y = 0; y < h; ++y) {
      std::memcpy(pixel.data() + y * rowStride,
          imgData + (h - y - 1) * rowStride,
          rowStride * sizeof(float));
    }
    return width > 0 && height > 0
        && (numComponents == 3 || numComponents == 4);
  }
#endif

  return false;
}

} // namespace tsd
