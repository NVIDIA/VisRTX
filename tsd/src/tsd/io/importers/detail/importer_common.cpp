// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/importer_common.hpp"
// tsd_core
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/Token.hpp"
// tsd_io
#include "tsd/io/importers.hpp"
// mikktspace
#include "mikktspace.h"
// anari
#include <anari/anari_cpp/ext/linalg.h>
// std
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>
#include <vector>

namespace tsd::io {

using namespace tsd::core;
using namespace tsd::scene;

// These two used to scan for one separator character themselves, which missed
// that Windows accepts '/' as well as '\\', and missed the bare filename --
// 'volume.raw' has no separator at all, and reporting no file for it turned
// every importer that guards on the result into a silent no-op for a path
// typed relative to the cwd.

// The file `filepath` names, without its directory. Empty only when the path
// names no file -- it is empty itself, or ends in a separator.
std::string fileOf(const std::string &filepath)
{
  return std::filesystem::path(filepath).filename().string();
}

// The directory `filepath` names, with the trailing separator kept so callers
// can concatenate a sibling file onto it. Empty when the path names no
// directory.
//
// Taken as the prefix fileOf() left behind rather than rebuilt from
// parent_path(), so the separator is the one the path already used and the two
// halves always rejoin into the original. Appending the platform's own
// separator instead would hand back '/a/b\\x.raw' for '/a/b/x.raw' on Windows.
std::string pathOf(const std::string &filepath)
{
  return filepath.substr(0, filepath.size() - fileOf(filepath).size());
}

std::string extensionOf(const std::string &filepath)
{
  size_t pos = filepath.rfind('.');
  if (pos == filepath.npos)
    return "";
  return filepath.substr(pos);
}

bool isAbsolute(const std::string &filepath)
{
  if (filepath.empty())
    return false;
#ifdef _WIN32
  // Drive letter (e.g. C:\) or UNC path (e.g. \\server\share)
  if (filepath.size() >= 3 && std::isalpha(filepath[0]) && filepath[1] == ':')
    return filepath[2] == '\\' || filepath[2] == '/';
  if (filepath.size() >= 2 && filepath[0] == '\\' && filepath[1] == '\\')
    return true;
#endif
  return filepath[0] == '/';
}

std::vector<std::string> splitString(const std::string &s, char delim)
{
  std::vector<std::string> result;
  std::istringstream stream(s);
  for (std::string token; std::getline(stream, token, delim);)
    result.push_back(token);
  return result;
}

tsd::scene::ArrayRef readArray(
    tsd::scene::Scene &scene, anari::DataType elementType, std::FILE *fp)
{
  tsd::scene::ArrayRef retval;

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

// Texture import shims ///////////////////////////////////////////////////////

// These forward to tsd::io::images, which owns decoding, orientation, keying,
// and lifetime for every image in the tree. They exist so the call sites that
// want the whole of it -- acquire, then build a Sampler for what came back --
// keep one signature. Like the makeImageSampler they end in, they take the
// ImageCache alone, so no caller can put the Sampler in a Scene the image
// never reached.

namespace {

ColorSpace colorSpaceOf(bool isLinear)
{
  return isLinear ? ColorSpace::LINEAR : ColorSpace::SRGB;
}

} // namespace

SamplerRef importTexture(ImageCache &cache,
    std::string filepath,
    bool isLinear,
    const SamplerSettings &settings)
{
  std::transform(
      filepath.begin(), filepath.end(), filepath.begin(), [](char c) {
        return c == '\\' ? '/' : c;
      });

  auto image = cache.acquire({filepath, colorSpaceOf(isLinear)});
  return makeImageSampler(cache, image, filepath, settings);
}

SamplerRef importTextureFromMemory(ImageCache &cache,
    const std::string &cacheKey,
    const std::string &displayName,
    const void *data,
    size_t numBytes,
    bool isLinear,
    const std::string &formatHint,
    const SamplerSettings &settings)
{
  auto image = cache.acquire(
      {cacheKey, colorSpaceOf(isLinear)}, data, numBytes, formatHint);
  return makeImageSampler(cache, image, displayName, settings);
}

SamplerRef importRawTexture2D(ImageCache &cache,
    const std::string &cacheKey,
    const std::string &displayName,
    const void *data,
    size_t width,
    size_t height,
    bool isLinear,
    const SamplerSettings &settings)
{
  auto image = cache.acquireDecoded({cacheKey, colorSpaceOf(isLinear)},
      isLinear ? ANARI_UFIXED8_VEC4 : ANARI_UFIXED8_RGBA_SRGB,
      width,
      height,
      RowOrder::TOP_DOWN,
      data);
  return makeImageSampler(cache, image, displayName, settings);
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
    size_t numVertices,
    bool flipTexCoordY,
    bool faceVaryingTangents)
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
    bool flipTexCoordY;
    bool faceVaryingTangents;
    size_t numIndices;
    size_t numVertices;
  } mesh;

  mesh.indices = indices;
  mesh.vertexPositions = vertexPositions;
  mesh.vertexNormals = vertexNormals;
  mesh.texCoords = texCoords;
  mesh.tangents = tangents;
  mesh.flipTexCoordY = flipTexCoordY;
  mesh.faceVaryingTangents = faceVaryingTangents;
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
    oc = {mesh->texCoords[vID].x,
        mesh->flipTexCoordY ? 1.0f - mesh->texCoords[vID].y
                            : mesh->texCoords[vID].y};
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
    if (mesh->faceVaryingTangents)
      vID = faceID * 3 + vertID;

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
    if (colors.size() < 2) {
      logError(
          "[import1dtTransferFunction] Expected at least two RGBA entries in "
          "file: %s",
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
  const std::string jsonContent{
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()};

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

          for (std::string opacityToken;
              std::getline(opacitySS, opacityToken, ',');) {
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
      opacityValues = {
          rgbValues[0], 0.0f, rgbValues[rgbValues.size() - 4], 1.0f};
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
      opacityPoints.push_back({opacityValues[i * 2], opacityValues[i * 2 + 1]});
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
      opacityPoints.insert(
          opacityPoints.begin(), {valueRange.lower, opacityPoints.front().y});
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

std::filesystem::path userColorMapDirectory()
{
#ifdef _WIN32
  if (const char *appData = std::getenv("APPDATA"); appData != nullptr)
    return std::filesystem::path(appData) / "tsd" / "colormaps";
#else
  if (const char *home = std::getenv("HOME"); home != nullptr)
    return std::filesystem::path(home) / ".config" / "tsd" / "colormaps";
#endif

  return std::filesystem::path("colormaps");
}

std::vector<UserColorMap> loadUserColorMaps()
{
  return loadUserColorMaps(userColorMapDirectory());
}

std::vector<UserColorMap> loadUserColorMaps(
    const std::filesystem::path &directory)
{
  namespace fs = std::filesystem;

  std::error_code ec;
  if (!fs::exists(directory, ec) || !fs::is_directory(directory, ec))
    return {};

  std::vector<fs::path> files;
  for (fs::directory_iterator it(directory, ec), end; !ec && it != end;
      it.increment(ec)) {
    const auto &entry = *it;
    if (!entry.is_regular_file(ec))
      continue;

    auto ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".1dt")
      files.push_back(entry.path());
  }

  if (ec) {
    logWarning("[loadUserColorMaps] Failed to scan directory '%s': %s",
        directory.string().c_str(),
        ec.message().c_str());
  }

  std::sort(
      files.begin(), files.end(), [](const fs::path &a, const fs::path &b) {
        return a.stem().string() < b.stem().string();
      });

  std::vector<UserColorMap> colorMaps;
  for (const auto &file : files) {
    auto tfn = importTransferFunction(file.string());
    if (tfn.colorPoints.size() < 2) {
      logWarning(
          "[loadUserColorMaps] Skipping color map '%s'", file.string().c_str());
      continue;
    }

    UserColorMap colorMap;
    colorMap.name = file.stem().string();
    colorMap.path = file;
    colorMap.colorPoints = std::move(tfn.colorPoints);

    auto existing = std::find_if(colorMaps.begin(),
        colorMaps.end(),
        [&](const UserColorMap &other) { return other.name == colorMap.name; });
    if (existing != colorMaps.end()) {
      logStatus("[loadUserColorMaps] Replaced color map '%s' from '%s'",
          colorMap.name.c_str(),
          colorMap.path.string().c_str());
      *existing = std::move(colorMap);
    } else {
      colorMaps.push_back(std::move(colorMap));
    }
  }

  return colorMaps;
}

#if TSD_USE_VTK
anari::DataType vtkTypeToANARIType(
    int vtkType, int numComps, const char *errorIdentifier)
{
  if (numComps > 4) {
    tsd::core::logError(
        "[%s] unsupported number of components %d", errorIdentifier, numComps);
    return ANARI_UNKNOWN;
  }

  numComps -= 1; // ANARI types are zero-indexed (e.g. FLOAT3 = FLOAT + 2)

  switch (vtkType) {
  case VTK_FLOAT:
    return ANARI_FLOAT32 + numComps;
  case VTK_DOUBLE:
    return ANARI_FLOAT64 + numComps;
  case VTK_CHAR:
    return ANARI_FIXED8 + numComps;
  case VTK_SHORT:
    return ANARI_FIXED16 + numComps;
  case VTK_INT:
    return ANARI_FIXED32 + numComps;
  case VTK_UNSIGNED_CHAR:
    return ANARI_UFIXED8 + numComps;
  case VTK_UNSIGNED_SHORT:
    return ANARI_UFIXED16 + numComps;
  case VTK_UNSIGNED_INT:
    return ANARI_UFIXED32 + numComps;
  default:
    tsd::core::logError(
        "[%s] unsupported vtk type %d[%d]", errorIdentifier, vtkType, numComps);
    return ANARI_UNKNOWN;
  }
}

tsd::scene::ArrayRef makeArray1DFromVTK(
    tsd::scene::Scene &scene, vtkDataArray *array, const char *errorIdentifier)
{
  const void *ptr = array->GetVoidPointer(0);
  const auto numTuples = array->GetNumberOfTuples();
  const int numComps = array->GetNumberOfComponents();
  const int vtkType = array->GetDataType();

  auto arr = scene.createArray(
      vtkTypeToANARIType(vtkType, numComps, errorIdentifier), numTuples);
  arr->setData(ptr);
  return arr;
}

tsd::scene::ArrayRef makeArray3DFromVTK(tsd::scene::Scene &scene,
    vtkDataArray *array,
    size_t w,
    size_t h,
    size_t d,
    const char *errorIdentifier)
{
  const void *ptr = array->GetVoidPointer(0);
  const int numComps = array->GetNumberOfComponents();
  const int vtkType = array->GetDataType();

  auto arr = scene.createArray(
      vtkTypeToANARIType(vtkType, numComps, errorIdentifier), w, h, d);
  arr->setData(ptr);
  return arr;
}
#endif

// Animation helpers ///////////////////////////////////////////////////////////

std::vector<float> makeLinearTimeBase(size_t count)
{
  std::vector<float> tb(count);
  float denom = count > 1 ? float(count - 1) : 1.f;
  for (size_t i = 0; i < count; i++)
    tb[i] = float(i) / denom;
  return tb;
}

void addValueTimeStepBindings(tsd::animation::Animation &anim,
    Object *target,
    const std::vector<Token> &paramNames,
    const std::vector<ObjectUsePtr<Array>> &dataArrays,
    const std::vector<float> &timeBase,
    tsd::animation::InterpolationRule interp)
{
  for (size_t i = 0; i < paramNames.size(); i++) {
    anim.addObjectParameterBinding(target,
        paramNames[i],
        dataArrays[i]->elementType(),
        dataArrays[i]->data(),
        timeBase.data(),
        timeBase.size(),
        interp);
  }
}

void addArrayTimeStepBindings(tsd::animation::Animation &anim,
    Object *target,
    const std::vector<Token> &paramNames,
    const std::vector<std::vector<ObjectUsePtr<Array>>> &arraysPerParam,
    const std::vector<float> &timeBase)
{
  for (size_t i = 0; i < paramNames.size(); i++) {
    auto &arrays = arraysPerParam[i];
    std::vector<Object *> objectPtrs(arrays.size());
    for (size_t j = 0; j < arrays.size(); j++)
      objectPtrs[j] = const_cast<Array *>(arrays[j].get());
    anim.addObjectParameterBinding(target,
        paramNames[i],
        ANARI_ARRAY1D,
        objectPtrs.data(),
        timeBase.data(),
        arrays.size(),
        tsd::animation::InterpolationRule::STEP);
  }
}

static math::float4 mat3ToQuat(
    math::float3 c0, math::float3 c1, math::float3 c2)
{
  // Shepperd's method
  float trace = c0.x + c1.y + c2.z;
  math::float4 q;
  if (trace > 0.f) {
    float s = 0.5f / std::sqrt(trace + 1.f);
    q = {(c1.z - c2.y) * s, (c2.x - c0.z) * s, (c0.y - c1.x) * s, 0.25f / s};
  } else if (c0.x > c1.y && c0.x > c2.z) {
    float s = 0.5f / std::sqrt(1.f + c0.x - c1.y - c2.z);
    q = {0.25f / s, (c0.y + c1.x) * s, (c2.x + c0.z) * s, (c1.z - c2.y) * s};
  } else if (c1.y > c2.z) {
    float s = 0.5f / std::sqrt(1.f + c1.y - c0.x - c2.z);
    q = {(c0.y + c1.x) * s, 0.25f / s, (c1.z + c2.y) * s, (c2.x - c0.z) * s};
  } else {
    float s = 0.5f / std::sqrt(1.f + c2.z - c0.x - c1.y);
    q = {(c2.x + c0.z) * s, (c1.z + c2.y) * s, 0.25f / s, (c0.y - c1.x) * s};
  }
  float len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  return {q.x / len, q.y / len, q.z / len, q.w / len};
}

void addTransformStepBinding(tsd::animation::Animation &anim,
    LayerNodeRef target,
    const std::vector<math::mat4> &frames,
    const std::vector<float> &timeBase)
{
  // Every array handed to the binding is sized by the decomposition below, so
  // the shorter of the two inputs is what can actually be read.
  size_t n = std::min(frames.size(), timeBase.size());
  std::vector<tsd::core::math::float4> rotation(n);
  std::vector<tsd::core::math::float3> translation(n);
  std::vector<tsd::core::math::float3> scale(n);

  for (size_t i = 0; i < n; i++) {
    auto &m = frames[i];
    math::float3 c0 = {m[0][0], m[0][1], m[0][2]};
    math::float3 c1 = {m[1][0], m[1][1], m[1][2]};
    math::float3 c2 = {m[2][0], m[2][1], m[2][2]};

    scale[i] = {length(c0), length(c1), length(c2)};
    if (scale[i].x > 0.f)
      c0 = c0 / scale[i].x;
    if (scale[i].y > 0.f)
      c1 = c1 / scale[i].y;
    if (scale[i].z > 0.f)
      c2 = c2 / scale[i].z;

    rotation[i] = mat3ToQuat(c0, c1, c2);
    translation[i] = {m[3][0], m[3][1], m[3][2]};
  }

  anim.addTransformBinding(target,
      timeBase.data(),
      rotation.data(),
      translation.data(),
      scale.data(),
      n);
}

} // namespace tsd::io
