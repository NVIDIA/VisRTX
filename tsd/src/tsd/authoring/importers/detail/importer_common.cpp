// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"
// stb_image
#include "tsd_stb/stb_image.h"
// std
#include <sstream>

namespace tsd {

#ifdef _WIN32
constexpr char path_sep = '\\';
#else
constexpr char path_sep = '/';
#endif

std::string pathOf(const std::string &filepath)
{
  size_t pos = filepath.find_last_of(path_sep);
  if (pos == std::string::npos)
    return "";
  return filepath.substr(0, pos + 1);
}

std::string fileOf(const std::string &filepath)
{
  size_t pos = filepath.find_last_of(path_sep);
  if (pos == std::string::npos)
    return "";
  return filepath.substr(pos + 1, filepath.size());
}

std::string extensionOf(const std::string &filepath)
{
  size_t pos = filepath.rfind('.');
  if (pos == filepath.npos)
    return "";
  return filepath.substr(pos);
}

std::vector<std::string> splitString(const std::string &s, char delim)
{
  std::vector<std::string> result;
  std::istringstream stream(s);
  for (std::string token; std::getline(stream, token, delim);)
    result.push_back(token);
  return result;
}

SamplerRef importTexture(
    Context &ctx, std::string filepath, TextureCache &cache)
{
  std::transform(
      filepath.begin(), filepath.end(), filepath.begin(), [](char c) {
        return c == '\\' ? '/' : c;
      });

  auto tex = cache[filepath];

  if (!tex) {
    int width, height, n;
    stbi_set_flip_vertically_on_load(1);
    void *data = stbi_loadf(filepath.c_str(), &width, &height, &n, 0);

    if (!data || n < 1) {
      if (!data) {
        logError(
            "[importTexture] failed to import texture '%s'", filepath.c_str());
      } else {
        logWarning("[importTexture] texture '%s' with %i channels not imported",
            filepath.c_str(),
            n);
      }
      return {};
    }

    tex = ctx.createObject<Sampler>(tokens::sampler::image2D);

    int texelType = ANARI_FLOAT32_VEC4;
    if (n == 3)
      texelType = ANARI_FLOAT32_VEC3;
    else if (n == 2)
      texelType = ANARI_FLOAT32_VEC2;
    else if (n == 1)
      texelType = ANARI_FLOAT32;

    auto dataArray = ctx.createArray(texelType, width, height);
    dataArray->setData(data);

    tex->setParameterObject("image"_t, *dataArray);
    tex->setParameter("inAttribute"_t, "attribute0");
    tex->setParameter("wrapMode1"_t, "repeat");
    tex->setParameter("wrapMode2"_t, "repeat");
    tex->setParameter("filter"_t, "linear");
    tex->setName(fileOf(filepath).c_str());

    cache[filepath] = tex;
  }

  return tex;
}

} // namespace tsd
