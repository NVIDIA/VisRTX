// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/io/importers/detail/dds.h"
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
// mikktspace
#include "mikktspace.h"
// stb_image
#include "stb_image.h"
#include "tsd/core/Token.hpp"
// std
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

SamplerRef importDdsTexture(
    Context &ctx, std::string filepath, TextureCache &cache)
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
      compressedFormat = alpha ? "BC1_RGBA"_t : "BC1_RGB"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC1_UNORM_SRGB: {
      // BC1: RGB/RGBA, 1bit alpha
      compressedFormat = alpha ? "BC1_RGBA_SRGB"_t : "BC1_RGB_SRGB"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC2_UNORM: {
      // BC2: RGB/RGBA, 4bit alpha
      compressedFormat = "BC2"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC2_UNORM_SRGB: {
      // BC2: RGB/RGBA, 4bit alpha
      compressedFormat = "BC2_SRGB"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC3_UNORM: {
      // BC3: RGB/RGBA, 8bit alpha
      compressedFormat = "BC3"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC3_UNORM_SRGB: {
      // BC3: RGB/RGBA, 8bit alpha
      compressedFormat = "BC3_SRGB"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC4_UNORM: {
      // BC4: R/RG
      compressedFormat = "BC4"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC4_SNORM: {
      // BC4: R/RG
      compressedFormat = "BC4_SNORM"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC5_UNORM: {
      // BC5: RG/RGBA
      compressedFormat = "BC5"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC5_SNORM: {
      // BC5: RG/RGBA
      compressedFormat = "BC5_SNORM"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC6H_UF16: {
      // BC6H: RGB
      compressedFormat = "BC6H_UFLOAT"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC6H_SF16: {
      // BC6H: RGB
      compressedFormat = "BC6H_SFLOAT"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC7_UNORM: {
      // BC7: RGB/RGBA
      compressedFormat = "BC7"_t;
      break;
    }
    case dds::DXGI_FORMAT_BC7_UNORM_SRGB: {
      // BC7: RGB/RGBA
      compressedFormat = "BC7_SRGB"_t;
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

      std::vector<std::byte> imageContent(linearSize);
      dds::vflipImage(dds, data(imageContent));

      dataArray = ctx.createArray(ANARI_INT8, linearSize);
      dataArray->setData(data(imageContent));
      dataArray->setMetadataValue("compressedFormat", compressedFormat.value());
      dataArray->setMetadataValue(
          "imageSize", U64Vec2(dds->header.width, dds->header.height));
    } else {
      logError("Unspported texture format for '%s'", filepath.c_str());
      return {};
    }
  }

  auto compressedFormat =
      dataArray->getMetadataValue("compressedFormat").get<std::string>();

  auto tex = ctx.createObject<Sampler>(tokens::sampler::compressedImage2D);
  tex->setParameterObject("image"_t, *dataArray);
  tex->setParameter("format"_t, compressedFormat.c_str());
  tex->setParameter(
      "size"_t, dataArray->getMetadataValue("imageSize").get<U64Vec2>());
  tex->setParameter("inAttribute"_t, "attribute0");
  tex->setParameter("wrapMode1"_t, "repeat");
  tex->setParameter("wrapMode2"_t, "repeat");
  tex->setParameter("filter"_t, "linear");
  tex->setName(fileOf(filepath).c_str());

  return tex;
}

SamplerRef importStbTexture(
    Context &ctx, std::string filepath, TextureCache &cache, bool isLinear)
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

    dataArray = ctx.createArray(texelType, width, height);
    dataArray->setData(data);

    stbi_image_free(data);
  }

  auto tex = ctx.createObject<Sampler>(tokens::sampler::image2D);

  tex->setParameterObject("image"_t, *dataArray);
  tex->setParameter("inAttribute"_t, "attribute0");
  tex->setParameter("wrapMode1"_t, "repeat");
  tex->setParameter("wrapMode2"_t, "repeat");
  tex->setParameter("filter"_t, "linear");
  tex->setName(fileOf(filepath).c_str());

  return tex;
}

SamplerRef importTexture(
    Context &ctx, std::string filepath, TextureCache &cache, bool isLinear)
{
  std::transform(
      filepath.begin(), filepath.end(), filepath.begin(), [](char c) {
        return c == '\\' ? '/' : c;
      });

  SamplerRef tex;
  if (filepath.size() > 4 && filepath.substr(filepath.size() - 4) == ".dds") {
    tex = importDdsTexture(ctx, filepath, cache);
  } else {
    tex = importStbTexture(ctx, filepath, cache, isLinear);
  }

  return tex;
}

SamplerRef makeDefaultColorMapSampler(Context &ctx, const float2 &range)
{
  auto samplerImageArray = ctx.createArray(ANARI_FLOAT32_VEC4, 3);
  auto *colorMapPtr = samplerImageArray->mapAs<math::float4>();
  colorMapPtr[0] = math::float4(0.f, 0.f, 1.f, 1.f);
  colorMapPtr[1] = math::float4(0.f, 1.f, 0.f, 1.f);
  colorMapPtr[2] = math::float4(1.f, 0.f, 0.f, 1.f);
  samplerImageArray->unmap();

  auto sampler = ctx.createObject<Sampler>(tokens::sampler::image1D);
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
    const float3 *texCoords,
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
    const float3 *texCoords;
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

} // namespace tsd::io
