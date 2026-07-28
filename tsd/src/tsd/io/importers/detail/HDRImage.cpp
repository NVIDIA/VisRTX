// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// std
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
// stb
#include "stb_image.h"
#include "stb_image_write.h"
#ifndef _WIN32
#include "tinyexr.h"
#endif
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/importers/detail/importer_common.hpp"

#include "HDRImage.h"

namespace tsd::io {

using namespace tsd::core;

#ifndef _WIN32
// Extract RGB pixel data from an EXRImage+EXRHeader into a flipped RGB float
// array. Returns false if no usable channels found.
static bool extractRGBFromEXRImage(const EXRImage &img,
    const EXRHeader &hdr,
    std::vector<float> &out_pixels)
{
  const int w = img.width;
  const int h = img.height;

  // Find R, G, B channel indices (also accept lowercase 'r','g','b')
  int idxR = -1, idxG = -1, idxB = -1;
  for (int i = 0; i < hdr.num_channels; i++) {
    std::string n = hdr.channels[i].name;
    if (n == "R" || n == "r")
      idxR = i;
    else if (n == "G" || n == "g")
      idxG = i;
    else if (n == "B" || n == "b")
      idxB = i;
  }

  // Fallback: use first channel(s) if standard names not found
  if (idxR == -1) {
    if (hdr.num_channels >= 3) {
      idxR = 0;
      idxG = 1;
      idxB = 2;
    } else if (hdr.num_channels >= 1) {
      idxR = idxG = idxB = 0; // grayscale
    } else {
      return false;
    }
  }
  if (idxG == -1)
    idxG = idxR;
  if (idxB == -1)
    idxB = idxR;

  // Only scanline format supported in this path (tiled multipart is uncommon
  // for HDRIs)
  if (img.images == nullptr) {
    logError(
        "[import_HDRI] tiled multipart EXR not supported, cannot load HDRI");
    return false;
  }

  out_pixels.resize(w * h * 3);
  const float *chanR = reinterpret_cast<const float *>(img.images[idxR]);
  const float *chanG = reinterpret_cast<const float *>(img.images[idxG]);
  const float *chanB = reinterpret_cast<const float *>(img.images[idxB]);

  for (int j = 0; j < h; j++) {
    for (int i = 0; i < w; i++) {
      int srcIdx = j * w + i;
      int dstIdx = 3 * ((h - j - 1) * w + i); // vertical flip
      out_pixels[dstIdx + 0] = chanR[srcIdx];
      out_pixels[dstIdx + 1] = chanG[srcIdx];
      out_pixels[dstIdx + 2] = chanB[srcIdx];
    }
  }
  return true;
}

// Load the first part of a multipart EXR into out_pixels (RGB, vertically
// flipped). Returns true on success.
static bool loadEXRMultipart(const char *filename,
    int &out_w,
    int &out_h,
    std::vector<float> &out_pixels)
{
  EXRVersion exr_version;
  if (ParseEXRVersionFromFile(&exr_version, filename) != TINYEXR_SUCCESS)
    return false;

  EXRHeader **exr_headers = nullptr;
  int num_parts = 0;
  const char *mp_err = nullptr;

  if (ParseEXRMultipartHeaderFromFile(
          &exr_headers, &num_parts, &exr_version, filename, &mp_err)
      != TINYEXR_SUCCESS) {
    if (mp_err)
      FreeEXRErrorMessage(mp_err);
    return false;
  }

  if (num_parts == 0) {
    free(exr_headers);
    return false;
  }

  // Convert HALF to FLOAT for all channels of the first part
  for (int i = 0; i < exr_headers[0]->num_channels; i++) {
    if (exr_headers[0]->pixel_types[i] == TINYEXR_PIXELTYPE_HALF)
      exr_headers[0]->requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
  }

  std::vector<EXRImage> exr_images(num_parts);
  for (auto &img : exr_images)
    InitEXRImage(&img);

  const char *load_err = nullptr;
  bool success = false;

  if (LoadEXRMultipartImageFromFile(exr_images.data(),
          (const EXRHeader **)exr_headers,
          static_cast<unsigned int>(num_parts),
          filename,
          &load_err)
      == TINYEXR_SUCCESS) {
    out_w = exr_images[0].width;
    out_h = exr_images[0].height;
    success =
        extractRGBFromEXRImage(exr_images[0], *exr_headers[0], out_pixels);
    if (!success)
      logError("[import_HDRI] could not extract RGB from multipart EXR: %s",
          filename);
  } else {
    logError("[import_HDRI] failed to load multipart EXR image: %s",
        load_err ? load_err : "unknown error");
    if (load_err)
      FreeEXRErrorMessage(load_err);
  }

  for (int p = 0; p < num_parts; p++) {
    FreeEXRImage(&exr_images[p]);
    FreeEXRHeader(exr_headers[p]);
    free(exr_headers[p]);
  }
  free(exr_headers);

  return success;
}
#endif // !_WIN32

bool HDRImage::import(std::string fileName)
{
  auto extension = extensionOf(fileName);

  if (extension != ".hdr" && extension != ".exr")
    return false;

  if (extension == ".hdr") {
    int w, h, n;
    // Bottom row first, matching the EXR branch below and the orientation a
    // Scene stores images in.
    stbi_set_flip_vertically_on_load(1);
    const float *imgData = stbi_loadf(fileName.c_str(), &w, &h, &n, STBI_rgb);
    stbi_set_flip_vertically_on_load(0); // this flag is global to stb
    width = w;
    height = h;
    numComponents = 3; // because of STBI_rgb
    if (width <= 0 || height <= 0 || n < 3) {
      stbi_image_free(const_cast<float *>(imgData));
      logError("[import_HDRI] error importing HDR image: %s", fileName.c_str());
      return false;
    }

    pixel.resize(w * h * 3);
    std::memcpy(pixel.data(), imgData, w * h * 3 * sizeof(float));
    stbi_image_free(const_cast<float *>(imgData));
    return true;
#ifdef _WIN32
  }
#else
  } else {
    int w = 0, h = 0;
    float *imgData = nullptr;
    const char *err = nullptr;
    int ret = LoadEXR(&imgData, &w, &h, fileName.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
      // Retry as multipart EXR (LoadEXR rejects multipart outright)
      if (loadEXRMultipart(fileName.c_str(), w, h, pixel)) {
        width = w;
        height = h;
        numComponents = 3;
        if (err)
          FreeEXRErrorMessage(err);
        return true;
      }
      logError("[import_HDRI] error importing EXR '%s': %s",
          fileName.c_str(),
          err ? err : "unknown error");
      if (err)
        FreeEXRErrorMessage(err);
      return false;
    }

    width = w;
    height = h;
    numComponents = 3;

    pixel.resize(w * h * 3);
    // LoadEXR returns RGBA; extract RGB and vertically flip.
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

} // namespace tsd::io
