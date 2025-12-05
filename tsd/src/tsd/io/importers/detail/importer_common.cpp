// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/importer_common.hpp"
#include <anari/anari_cpp/ext/linalg.h>
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/importers/detail/dds.h"
// mikktspace
#include "mikktspace.h"
// stb_image
#include "stb_image.h"
#include "tsd/core/Token.hpp"
// std
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <sstream>

using U64Vec2 = tsd::math::vec<std::uint64_t, 2>;
namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(U64Vec2, ANARI_UINT64_VEC2);
}

namespace tsd::io {

using namespace tsd::core;

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

tsd::core::ArrayRef readArray(
    tsd::core::Scene &scene, anari::DataType elementType, std::FILE *fp)
{
  tsd::core::ArrayRef retval;

  size_t size = 0;
  auto r = std::fread(&size, sizeof(size_t), 1, fp);

  if (size > 0) {
    retval = scene.createArray(elementType, size);
    auto *dst = retval->map();
    r = std::fread(dst, anari::sizeOf(elementType), size, fp);
    retval->unmap();
  }

  return retval;
}

SamplerRef importDdsTexture(
    Scene &scene, std::string filepath, TextureCache &cache)
{
  auto dataArray = cache[filepath];
  if (!dataArray.valid()) {
    std::ifstream ifs(filepath, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
      logError("[importDdsTexture] failed to open file '%s'", filepath.c_str());
      return {};
    }

    std::vector<char> buffer((std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>());
    auto dds = reinterpret_cast<const dds::DdsFile *>(data(buffer));
    if (dds->magic != dds::DDS_MAGIC
        || dds->header.size != sizeof(dds::DdsHeader)) {
      logError("[importDdsTexture] invalid DDS file '%s'", filepath.c_str());
      return {};
    }

    // Check if we have a dxt10 header
    constexpr const auto baseReqFlags = dds::DDSD_CAPS | dds::DDSD_HEIGHT
        | dds::DDSD_WIDTH | dds::DDSD_PIXELFORMAT;
    if ((dds->header.flags & baseReqFlags) != baseReqFlags) {
      logError("[importDdsTexture] invalid DDS file '%s'", filepath.c_str());
      return {};
    }

    constexpr const auto textureReqFlags = dds::DDSCAPS_TEXTURE;
    if ((dds->header.caps & textureReqFlags) != textureReqFlags) {
      logError("[importDdsTexture] invalid DDS file '%s'", filepath.c_str());
      return {};
    }

    Token compressedFormat = {};
    Token format = {};
    bool alpha = dds->header.pixelFormat.flags & dds::DDPF_ALPHAPIXELS;
    switch (dds::getDxgiFormat(dds)) {
    case dds::DXGI_FORMAT_BC1_UNORM: {
      // BC1: RGB/RGBA, 1bit alpha
      compressedFormat = alpha ? "BC1_RGBA" : "BC1_RGB";
      break;
    }
    case dds::DXGI_FORMAT_BC1_UNORM_SRGB: {
      // BC1: RGB/RGBA, 1bit alpha
      compressedFormat = alpha ? "BC1_RGBA_SRGB" : "BC1_RGB_SRGB";
      break;
    }
    case dds::DXGI_FORMAT_BC2_UNORM: {
      // BC2: RGB/RGBA, 4bit alpha
      compressedFormat = "BC2";
      break;
    }
    case dds::DXGI_FORMAT_BC2_UNORM_SRGB: {
      // BC2: RGB/RGBA, 4bit alpha
      compressedFormat = "BC2_SRGB";
      break;
    }
    case dds::DXGI_FORMAT_BC3_UNORM: {
      // BC3: RGB/RGBA, 8bit alpha
      compressedFormat = "BC3";
      break;
    }
    case dds::DXGI_FORMAT_BC3_UNORM_SRGB: {
      // BC3: RGB/RGBA, 8bit alpha
      compressedFormat = "BC3_SRGB";
      break;
    }
    case dds::DXGI_FORMAT_BC4_UNORM: {
      // BC4: R/RG
      compressedFormat = "BC4";
      break;
    }
    case dds::DXGI_FORMAT_BC4_SNORM: {
      // BC4: R/RG
      compressedFormat = "BC4_SNORM";
      break;
    }
    case dds::DXGI_FORMAT_BC5_UNORM: {
      // BC5: RG/RGBA
      compressedFormat = "BC5";
      break;
    }
    case dds::DXGI_FORMAT_BC5_SNORM: {
      // BC5: RG/RGBA
      compressedFormat = "BC5_SNORM";
      break;
    }
    case dds::DXGI_FORMAT_BC6H_UF16: {
      // BC6H: RGB
      compressedFormat = "BC6H_UFLOAT";
      break;
    }
    case dds::DXGI_FORMAT_BC6H_SF16: {
      // BC6H: RGB
      compressedFormat = "BC6H_SFLOAT";
      break;
    }
    case dds::DXGI_FORMAT_BC7_UNORM: {
      // BC7: RGB/RGBA
      compressedFormat = "BC7";
      break;
    }
    case dds::DXGI_FORMAT_BC7_UNORM_SRGB: {
      // BC7: RGB/RGBA
      compressedFormat = "BC7_SRGB";
      break;
    }

    default: {
      logError(
          "[importDdsTexture] unsupported DDS format '%c%c%c%c' for file '%s'",
          dds->header.pixelFormat.fourCC & 0xff,
          (dds->header.pixelFormat.fourCC >> 8) & 0xff,
          (dds->header.pixelFormat.fourCC >> 16) & 0xff,
          (dds->header.pixelFormat.fourCC >> 24) & 0xff,
          filepath.c_str());
      break;
    }
    }

    if (compressedFormat) {
      // Simple  implementation that only handling single level mipmaps
      // and non cubemap textures.
      auto linearSize = dds::computeLinearSize(dds);

      if ((dds->header.flags & dds::DDSD_LINEARSIZE)
          && (linearSize != dds->header.pitchOrLinearSize)) {
        logError(
            "[importDdsTexture] ignoring invalid linear size %u (should be %u) for compressed texture '%s'",
            dds->header.pitchOrLinearSize,
            linearSize,
            filepath.c_str());
      }

      dataArray = scene.createArray(ANARI_INT8, linearSize);
      dataArray->setData(dds::getDataPointer(dds));
      dataArray->setMetadataValue("compressedFormat", compressedFormat.value());
      dataArray->setMetadataValue(
          "imageSize", U64Vec2(dds->header.width, dds->header.height));
    } else {
      logError("Unspported texture format for '%s'", filepath.c_str());
      return {};
    }
  }

  auto compressedFormat =
      dataArray->getMetadataValue("compressedFormat").getString();

  auto tex = scene.createObject<Sampler>(tokens::sampler::compressedImage2D);
  tex->setParameterObject("image", *dataArray);
  tex->setParameter("format", compressedFormat.c_str());
  tex->setParameter(
      "size", dataArray->getMetadataValue("imageSize").get<U64Vec2>());
  tex->setParameter("inAttribute", "attribute0");
  tex->setParameter("wrapMode1", "repeat");
  tex->setParameter("wrapMode2", "repeat");
  tex->setParameter("filter", "linear");
  tex->setName(fileOf(filepath).c_str());

  return tex;
}

SamplerRef importStbTexture(
    Scene &scene, std::string filepath, TextureCache &cache, bool isLinear)
{
  auto dataArray = cache[filepath];
  if (!dataArray.valid()) {
    int width, height, n;
    if (isLinear) {
      stbi_ldr_to_hdr_scale(1.0f);
      stbi_ldr_to_hdr_gamma(1.0f);
    } else {
      stbi_ldr_to_hdr_scale(1.0f);
      stbi_ldr_to_hdr_gamma(2.2f);
    }
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

    int texelType = ANARI_FLOAT32_VEC4;
    if (n == 3)
      texelType = ANARI_FLOAT32_VEC3;
    else if (n == 2)
      texelType = ANARI_FLOAT32_VEC2;
    else if (n == 1)
      texelType = ANARI_FLOAT32;

    dataArray = scene.createArray(texelType, width, height);
    dataArray->setData(data);

    stbi_image_free(data);
  }

  auto tex = scene.createObject<Sampler>(tokens::sampler::image2D);

  tex->setParameterObject("image", *dataArray);
  tex->setParameter("inAttribute", "attribute0");
  tex->setParameter("wrapMode1", "repeat");
  tex->setParameter("wrapMode2", "repeat");
  tex->setParameter("filter", "linear");
  tex->setName(fileOf(filepath).c_str());

  return tex;
}

SamplerRef importTexture(
    Scene &scene, std::string filepath, TextureCache &cache, bool isLinear)
{
  std::transform(
      filepath.begin(), filepath.end(), filepath.begin(), [](char c) {
        return c == '\\' ? '/' : c;
      });

  SamplerRef tex;
  if (filepath.size() > 4 && filepath.substr(filepath.size() - 4) == ".dds") {
    tex = importDdsTexture(scene, filepath, cache);
  } else {
    tex = importStbTexture(scene, filepath, cache, isLinear);
  }

  return tex;
}

SamplerRef makeDefaultColorMapSampler(Scene &scene, const float2 &range)
{
  auto samplerImageArray = scene.createArray(ANARI_FLOAT32_VEC4, 3);
  auto *colorMapPtr = samplerImageArray->mapAs<math::float4>();
  colorMapPtr[0] = math::float4(0.f, 0.f, 1.f, 1.f);
  colorMapPtr[1] = math::float4(0.f, 1.f, 0.f, 1.f);
  colorMapPtr[2] = math::float4(1.f, 0.f, 0.f, 1.f);
  samplerImageArray->unmap();

  auto sampler = scene.createObject<Sampler>(tokens::sampler::image1D);
  sampler->setParameter("inAttribute", "attribute0");
  sampler->setParameter("inTransform", tsd::math::float2(range.x, range.y))
      ->setUsage(ParameterUsageHint::VALUE_RANGE_TRANSFORM);
  sampler->setParameter("filter", "linear");
  sampler->setParameter("wrapMode", "mirrorRepeat");
  sampler->setParameterObject("image", *samplerImageArray);

  return sampler;
}

bool calcTangentsForTriangleMesh(const uint3 *indices,
    const float3 *vertexPositions,
    const float3 *vertexNormals,
    const float2 *texCoords,
    float4 *tangents,
    size_t numIndices,
    size_t numVertices)
{
  if (!texCoords)
    return false;

  SMikkTSpaceInterface iface{};
  SMikkTSpaceContext context{};

  struct Mesh
  {
    const uint3 *indices;
    const float3 *vertexPositions;
    const float3 *vertexNormals;
    const float2 *texCoords;
    float4 *tangents;
    size_t numIndices;
    size_t numVertices;
  } mesh;

  mesh.indices = indices;
  mesh.vertexPositions = vertexPositions;
  mesh.vertexNormals = vertexNormals;
  mesh.texCoords = texCoords;
  mesh.tangents = tangents;
  mesh.numIndices = numIndices;
  mesh.numVertices = numVertices;

  // callback to get num faces of mesh
  iface.m_getNumFaces = [](const SMikkTSpaceContext *ctx) -> int {
    Mesh *mesh = (Mesh *)ctx->m_pUserData;
    return (int)mesh->numIndices;
  };

  // callback to get num verts of a single face (hardcoded to 3 for triangles)
  iface.m_getNumVerticesOfFace = [](const SMikkTSpaceContext *ctx,
                                     const int faceID) -> int {
    (void)ctx;
    (void)faceID;
    return 3;
  };

  // callback to get the vertex normal
  iface.m_getNormal = [](const SMikkTSpaceContext *ctx,
                          float *outnormal,
                          const int faceID,
                          const int vertID) {
    Mesh *mesh = (Mesh *)ctx->m_pUserData;

    float3 &on = (float3 &)*outnormal;

    uint3 index = mesh->indices[faceID];

    if (mesh->vertexNormals) {
      unsigned vID = index[vertID];
      on = mesh->vertexNormals[vID];
    } else {
      float3 v1 = mesh->vertexPositions[index.x];
      float3 v2 = mesh->vertexPositions[index.y];
      float3 v3 = mesh->vertexPositions[index.z];
      on = normalize(cross(v2 - v1, v3 - v1));
    }
  };

  iface.m_getPosition = [](const SMikkTSpaceContext *ctx,
                            float *outpos,
                            const int faceID,
                            const int vertID) {
    Mesh *mesh = (Mesh *)ctx->m_pUserData;

    float3 &op = (float3 &)*outpos;

    uint3 index = mesh->indices[faceID];

    unsigned vID = index[vertID];
    op = mesh->vertexPositions[vID];
  };

  // callback to get the texture coordinate (the mesh *must* have these!)
  iface.m_getTexCoord = [](const SMikkTSpaceContext *ctx,
                            float *outcoord,
                            const int faceID,
                            const int vertID) {
    Mesh *mesh = (Mesh *)ctx->m_pUserData;

    float2 &oc = (float2 &)*outcoord;

    uint3 index = mesh->indices[faceID];

    assert(mesh->texCoords);
    unsigned vID = index[vertID];
    oc = {mesh->texCoords[vID].x, 1.0f - mesh->texCoords[vID].y};
  };

  // callback to assign output tangents
  iface.m_setTSpaceBasic = [](const SMikkTSpaceContext *ctx,
                               const float *tangentVector,
                               const float tangentSign,
                               const int faceID,
                               const int vertID) {
    Mesh *mesh = (Mesh *)ctx->m_pUserData;

    uint3 index = mesh->indices[faceID];

    unsigned vID = index[vertID];

    float4 &outtangent = mesh->tangents[vID];

    outtangent.x = tangentVector[0];
    outtangent.y = tangentVector[1];
    outtangent.z = tangentVector[2];
    outtangent.w = tangentSign;
  };

  context.m_pInterface = &iface;
  context.m_pUserData = &mesh;

  return genTangSpaceDefault(&context);
}

// Transfer function import functions

static core::TransferFunction import1dtTransferFunction(
    const std::string &filepath)
{
  if (std::ifstream file(filepath); !file.is_open()) {
    logError("[import1dtTransferFunction] Failed to open file: %s",
        filepath.c_str());
    return {};
  } else {
    std::vector<core::ColorPoint> colors;
    std::vector<core::OpacityPoint> opacities;

    // Read all RGBA lines
    for (std::string line; std::getline(file, line);) {
      // Skip empty lines and comments
      if (line.empty() || line[0] == '#')
        continue;

      std::istringstream iss(line);
      float r, g, b, a;

      if (!(iss >> r >> g >> b >> a)) {
        logWarning(
            "[import1dtTransferFunction] Failed to parse line, skipping");
        continue;
      }

      const auto idx = static_cast<float>(colors.size());
      colors.push_back({idx, r, g, b});
      opacities.push_back({idx, a});
    }

    if (colors.empty()) {
      logError(
          "[import1dtTransferFunction] No valid RGBA entries found in file: %s",
          filepath.c_str());
      return {};
    }

    const float normalizer = 1.0f / static_cast<float>(colors.size() - 1);
    for (auto &c : colors)
      c.x *= normalizer;
    for (auto &o : opacities)
      o.x *= normalizer;

    return {colors, opacities};
  }
}

static core::TransferFunction importParaViewTransferFunction(
    const std::string &filepath)
{
  std::ifstream file(filepath);
  if (!file.is_open()) {
    logError("[importParaViewTransferFunction] Failed to open file: %s",
        filepath.c_str());
    return {};
  }

  // Read entire file
  const std::string jsonContent{(
      std::istreambuf_iterator<char>(file)),
      std::istreambuf_iterator<char>()};

  // Parse RGBPoints array
  if (const auto rgbPointsPos = jsonContent.find("\"RGBPoints\"");
      rgbPointsPos == std::string::npos) {
    logError("[importParaViewTransferFunction] No RGBPoints found in file: %s",
        filepath.c_str());
    return {};
  } else if (const auto arrayStart = jsonContent.find("[", rgbPointsPos);
             arrayStart == std::string::npos) {
    logError(
        "[importParaViewTransferFunction] Invalid RGBPoints format in file: %s",
        filepath.c_str());
    return {};
  } else {

    int bracketCount = 0;
    size_t arrayEnd = arrayStart;
    for (size_t i = arrayStart; i < jsonContent.length(); ++i) {
      if (jsonContent[i] == '[')
        bracketCount++;
      else if (jsonContent[i] == ']') {
        bracketCount--;
        if (bracketCount == 0) {
          arrayEnd = i;
          break;
        }
      }
    }

    if (arrayEnd == arrayStart) {
      logError(
          "[importParaViewTransferFunction] Could not find end of RGBPoints in file: %s",
          filepath.c_str());
      return {};
    }

    // Parse RGBPoints values
    const auto arrayContent =
        jsonContent.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
    std::vector<float> rgbValues;
    std::istringstream ss(arrayContent);

    for (std::string token; std::getline(ss, token, ',');) {
      // Trim whitespace
      if (const auto first = token.find_first_not_of(" \t\n\r");
          first != std::string::npos) {
        const auto last = token.find_last_not_of(" \t\n\r");
        token = token.substr(first, last - first + 1);

        try {
          rgbValues.push_back(std::stof(token));
        } catch (const std::exception &) {
          logError(
              "[importParaViewTransferFunction] Invalid RGBPoints value '%s' in file: %s",
              token.c_str(),
              filepath.c_str());
        }
      }
    }

    // RGBPoints format: [dataValue, r, g, b, dataValue, r, g, b, ...]
    if (rgbValues.size() % 4 != 0 || rgbValues.empty()) {
      logError(
          "[importParaViewTransferFunction] Invalid RGBPoints data in file: %s",
          filepath.c_str());
      return {};
    }

    // Parse optional Points array for opacity
    std::vector<float> opacityValues;
    if (const auto pointsPos = jsonContent.find("\"Points\"");
        pointsPos != std::string::npos) {
      if (const auto opacityArrayStart = jsonContent.find("[", pointsPos);
          opacityArrayStart != std::string::npos) {
        int opacityBracketCount = 0;
        size_t opacityArrayEnd = opacityArrayStart;
        for (size_t i = opacityArrayStart; i < jsonContent.length(); ++i) {
          if (jsonContent[i] == '[')
            opacityBracketCount++;
          else if (jsonContent[i] == ']') {
            opacityBracketCount--;
            if (opacityBracketCount == 0) {
              opacityArrayEnd = i;
              break;
            }
          }
        }

        if (opacityArrayEnd != opacityArrayStart) {
          const auto opacityContent = jsonContent.substr(
              opacityArrayStart + 1, opacityArrayEnd - opacityArrayStart - 1);
          std::istringstream opacitySS(opacityContent);

          for (std::string opacityToken; std::getline(opacitySS, opacityToken, ',');) {
            // Trim whitespace
            if (const auto first = opacityToken.find_first_not_of(" \t\n\r");
                first != std::string::npos) {
              const auto last = opacityToken.find_last_not_of(" \t\n\r");
              opacityToken = opacityToken.substr(first, last - first + 1);

              try {
                opacityValues.push_back(std::stof(opacityToken));
              } catch (const std::exception &) {
                // Skip invalid values
              }
            }
          }
        }
      }
    }

    // Points format: [dataValue, alpha, dataValue, alpha, ...]
    if (!opacityValues.empty()
        && (opacityValues.size() % 2 != 0 || opacityValues[0] != rgbValues[0]
            || opacityValues[opacityValues.size() - 2]
                != rgbValues[rgbValues.size() - 4])) {
      logError(
          "[importParaViewTransferFunction] Invalid Points data in file: %s, ignoring opacity",
          filepath.c_str());
      // Build a simple opacity ramp
      opacityValues = {rgbValues[0], 0.0f, rgbValues[rgbValues.size() - 4], 1.0f};
    }

    std::vector<ColorPoint> colorPoints;
    const size_t numRGBPoints = rgbValues.size() / 4;
    colorPoints.reserve(numRGBPoints);
    for (size_t i = 0; i < numRGBPoints; ++i) {
      colorPoints.push_back({rgbValues[i * 4],
          rgbValues[i * 4 + 1],
          rgbValues[i * 4 + 2],
          rgbValues[i * 4 + 3]});
    }

    std::vector<OpacityPoint> opacityPoints;
    const size_t numOpacityPoints = opacityValues.size() / 2;
    opacityPoints.reserve(numOpacityPoints);
    for (size_t i = 0; i < numOpacityPoints; ++i) {
      opacityPoints.push_back(
          {opacityValues[i * 2], opacityValues[i * 2 + 1]});
    }

    const auto valueRange =
        math::box1(std::min(colorPoints.front().x, opacityPoints.front().x),
            std::max(colorPoints.back().x, opacityPoints.back().x));

    // Make sure the extreme points are defined for 0 and 1
    if (valueRange.lower < colorPoints.front().x) {
      const auto &front = colorPoints.front();
      colorPoints.insert(
          colorPoints.begin(), {valueRange.lower, front.y, front.z, front.w});
    }
    if (valueRange.upper > colorPoints.back().x) {
      const auto &back = colorPoints.back();
      colorPoints.push_back({valueRange.upper, back.y, back.z, back.w});
    }
    if (valueRange.lower < opacityPoints.front().x) {
      opacityPoints.insert(opacityPoints.begin(),
          {valueRange.lower, opacityPoints.front().y});
    }
    if (valueRange.upper > opacityPoints.back().x) {
      opacityPoints.push_back({valueRange.upper, opacityPoints.back().y});
    }

    // And normalize to [0, 1]
    const float normalizer = 1.0f / (valueRange.upper - valueRange.lower);
    for (auto &c : colorPoints)
      c.x = (c.x - valueRange.lower) * normalizer;
    for (auto &o : opacityPoints)
      o.x = (o.x - valueRange.lower) * normalizer;

    return {colorPoints, opacityPoints, valueRange};
  }
}

core::TransferFunction importTransferFunction(const std::string &filepath)
{
  auto ext = extensionOf(filepath);

  // Convert extension to lowercase for comparison
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

  if (ext == ".1dt") {
    return import1dtTransferFunction(filepath);
  } else if (ext == ".json") {
    return importParaViewTransferFunction(filepath);
  }

  logError(
      "[importTransferFunction] Unsupported file extension: %s", ext.c_str());
  return {};
}

} // namespace tsd::io
