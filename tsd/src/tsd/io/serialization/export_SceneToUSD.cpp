// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#ifndef TSD_USE_USD
#define TSD_USE_USD 1
#endif

// tsd
#include "tsd/core/Logging.hpp"
#include "tsd/io/serialization.hpp"

// anari
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/frontend/anari_enums.h>
#include <anari/frontend/type_utility.h>

#include "tsd/core/scene/Scene.hpp"
#include "tsd/core/scene/objects/Array.hpp"
#include "tsd/core/scene/objects/Camera.hpp"
#include "tsd/core/scene/objects/Material.hpp"
#include "tsd/core/scene/objects/Sampler.hpp"
#include "tsd/core/scene/objects/Surface.hpp"

#if TSD_USE_USD
// stb and tinyexr
#include "stb_image.h"
#include "stb_image_write.h"
#include "tinyexr.h"

// usd
#include <pxr/base/gf/declare.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformOp.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>

// std
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std::string_literals;
using namespace std::chrono_literals;

namespace tsd::io {

static const auto allObjectsPath = pxr::SdfPath("/AllObjects");
static const auto allLightsPath =
    allObjectsPath.AppendChild(pxr::TfToken("Lights"));
static const auto allSurfacesPath =
    allObjectsPath.AppendChild(pxr::TfToken("Surfaces"));
static const auto allMaterialsPath =
    allObjectsPath.AppendChild(pxr::TfToken("Materials"));
static const auto allLayersPath = pxr::SdfPath("/AllLayers");

// Some global state
static std::filesystem::path outputPath;
static std::unordered_map<const Object *, pxr::SdfPath> objectToUsdPathMap;
static std::unordered_map<const Sampler *, std::filesystem::path>
    samplerToFilepathMap;

static std::unordered_set<std::filesystem::path> usedTextureFileNames;

static std::vector<std::future<std::string>> textureWriteQueue;

static std::string sanitizeSamplerName(const std::string &samplerName)
{
  std::string baseName = samplerName;
  // Replace spaces and special characters with underscores
  std::replace_if(
      baseName.begin(),
      baseName.end(),
      [](char c) { return !std::isalnum(c) && c != '_'; },
      '_');

  // Ensure the name is not empty
  if (baseName.empty()) {
    baseName = "sampler";
  }

  // Ensure uniqueness by appending a number if necessary
  std::string uniqueName;
  int uniqueIndex = 0;
  do {
    uniqueName = baseName + "_" + std::to_string(uniqueIndex++);
  } while (usedTextureFileNames.count(uniqueName));

  usedTextureFileNames.insert(uniqueName);

  return uniqueName;
}

pxr::SdfPath allocateUniquePath(
    pxr::UsdStagePtr stage, const pxr::SdfPath &path)
{
  const auto basePath = path.GetParentPath();
  auto baseName = path.GetName();

  int uniqueIndex = 0;
  auto newPath = pxr::SdfPath::EmptyPath();
  do {
    auto uniqueName = baseName + "_" + std::to_string(uniqueIndex++);
    newPath = basePath.AppendChild(pxr::TfToken(uniqueName.c_str()));
  } while (stage->GetObjectAtPath(newPath));

  return newPath;
}

static std::filesystem::path tsdSamplerToFile(
    pxr::UsdStageRefPtr &stage, const Sampler *sampler)
{
  assert(sampler);

  if (auto it = samplerToFilepathMap.find(sampler);
      it != samplerToFilepathMap.end()) {
    return it->second;
  }

  auto outputBasePath = outputPath / "textures";
  std::filesystem::create_directories(outputBasePath);

  if (sampler->name().empty()) {
    outputBasePath /= std::to_string(reinterpret_cast<std::uintptr_t>(sampler));
  } else {
    outputBasePath /= sampler->name();
  }

  // File name sanitization handles uniqueness
  outputBasePath = outputBasePath.replace_filename(
      sanitizeSamplerName(outputBasePath.filename().string()));

  if (sampler->subtype() == tokens::sampler::image2D) {
    auto image = sampler->parameterValueAsObject<Array>("image");
    auto width = image->dim(0);
    auto height = image->dim(1);
    auto format = image->elementType();

    if (format == ANARI_FLOAT32_VEC3 || format == ANARI_FLOAT32_VEC4) {
      // Write as EXR for HDR data
      outputBasePath += ".exr";
      int channels = (format == ANARI_FLOAT32_VEC3) ? 3 : 4;

      textureWriteQueue.push_back(std::async([=]() -> std::string {
        const char *err = nullptr;
        int ret = SaveEXR(reinterpret_cast<const float *>(image->data()),
            width,
            height,
            channels,
            0,
            outputBasePath.string().c_str(),
            &err);
        if (ret != TINYEXR_SUCCESS && err) {
          auto result =
              "Failed to save "s + outputBasePath.string() + " : " + err;
          FreeEXRErrorMessage(err);
          return result;
        } else {
          return "Saved "s + outputBasePath.string();
        }
      }));
    } else {
      // Write as PNG for LDR data
      outputBasePath += ".png";
      int channels = 0;
      if (format == ANARI_UFIXED8_VEC3 || format == ANARI_UFIXED8_RGB_SRGB)
        channels = 3;
      else if (format == ANARI_UFIXED8_VEC4
          || format == ANARI_UFIXED8_RGBA_SRGB)
        channels = 4;
      else if (format == ANARI_UFIXED8 || format == ANARI_UFIXED8_R_SRGB)
        channels = 1;

      if (channels > 0) {
        textureWriteQueue.push_back(std::async([=]() -> std::string {
          int ret = stbi_write_png(outputBasePath.string().c_str(),
              width,
              height,
              channels,
              image->data(),
              width * channels);
          if (!ret) {
            return "Failed to save "s + outputBasePath.string();
          } else {
            return "Saved "s + outputBasePath.string();
          }
        }));
      }
    }
  } else if (sampler->subtype() == tokens::sampler::compressedImage2D) {
    auto data = sampler->parameterValueAsObject<Array>("image");
    auto ddsFile =
        std::ofstream(outputBasePath.string() + ".dds", std::ios::binary);
    ddsFile.write(reinterpret_cast<const char *>(data->data()),
        data->size() * data->elementSize());
    ddsFile.close();
  }

  samplerToFilepathMap.insert({sampler, outputBasePath});
  return outputBasePath;
}

static pxr::TfToken tsdSamplertoScalarOutputToken(const Sampler *sampler)
{
  assert(sampler);
  auto transform = sampler->parameterValueAs<math::mat4>("outTransform")
                       .value_or(mat4(math::identity));
  auto x = transform.row(0);
  switch (argmax(x)) {
  case 0:
    return pxr::TfToken("r");
  case 1:
    return pxr::TfToken("g");
  case 2:
    return pxr::TfToken("b");
  case 3:
    return pxr::TfToken("a");
  default:
    return pxr::TfToken("r");
  }
}

// Helper function to create a texture shader and connect it to a material input
static pxr::UsdShadeShader tsdSamplertoUSD(pxr::UsdStageRefPtr &stage,
    const pxr::SdfPath &materialPath,
    const std::string &inputName,
    const Sampler *sampler,
    bool normalMap = false)
{
  assert(sampler);

  // Export the texture file
  auto textureFile = tsdSamplerToFile(stage, sampler);

  // Create UsdUVTexture shader
  auto textureName = inputName + "Texture";
  pxr::SdfPath texturePath =
      materialPath.AppendChild(pxr::TfToken(textureName.c_str()));
  auto textureShader = pxr::UsdShadeShader::Define(stage, texturePath);
  textureShader.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdUVTexture")));

  // Set the file path
  textureShader.CreateInput(pxr::TfToken("file"), pxr::SdfValueTypeNames->Asset)
      .Set(pxr::SdfAssetPath(textureFile.string()));

  // Set texture wrapping (assuming repeat for now)
  textureShader
      .CreateInput(pxr::TfToken("wrapS"), pxr::SdfValueTypeNames->Token)
      .Set(pxr::TfToken("repeat"));
  textureShader
      .CreateInput(pxr::TfToken("wrapT"), pxr::SdfValueTypeNames->Token)
      .Set(pxr::TfToken("repeat"));

  if (normalMap) {
    textureShader
        .CreateInput(pxr::TfToken("scale"), pxr::SdfValueTypeNames->Float4)
        .Set(pxr::GfVec4f(1.0f, -1.0f, 1.0f, 1.0f));
  }

  // Create primvar reader for UV coordinates
  auto primvarName = inputName + "PrimvarReader";
  auto primvarPath =
      materialPath.AppendChild(pxr::TfToken(primvarName.c_str()));
  auto primvarReader = pxr::UsdShadeShader::Define(stage, primvarPath);
  primvarReader.CreateIdAttr(
      pxr::VtValue(pxr::TfToken("UsdPrimvarReader_float2")));
  auto attributeName = sampler->parameterValueAs<std::string>("inAttribute")
                           .value_or("attribute0");
  primvarReader
      .CreateInput(pxr::TfToken("varname"), pxr::SdfValueTypeNames->Token)
      .Set(pxr::TfToken(attributeName));

  // Going from top-down to bottom-up.
  auto transformName = inputName + "Transform2d";
  auto transformPath =
      materialPath.AppendChild(pxr::TfToken(transformName.c_str()));
  auto transform = pxr::UsdShadeShader::Define(stage, transformPath);
  transform.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdTransform2d")));
  transform.CreateInput(pxr::TfToken("scale"), pxr::SdfValueTypeNames->Float2)
      .Set(pxr::GfVec2f(1.0f, -1.0f));
  transform
      .CreateInput(pxr::TfToken("translation"), pxr::SdfValueTypeNames->Float2)
      .Set(pxr::GfVec2f(0.0f, 1.0f));
  auto transformInput =
      transform.CreateInput(pxr::TfToken("in"), pxr::SdfValueTypeNames->Float2);
  transformInput.ConnectToSource(
      primvarReader.ConnectableAPI(), pxr::TfToken("result"));

  // Connect primvar reader to texture shader
  auto textureStInput = textureShader.CreateInput(
      pxr::TfToken("st"), pxr::SdfValueTypeNames->Float2);
  textureStInput.ConnectToSource(
      transform.ConnectableAPI(), pxr::TfToken("result"));

  return textureShader;
}

static pxr::SdfPath tsdMaterialToPreviewSurfaceUSD(
    pxr::UsdStageRefPtr &stage, const Material *material)
{
  if (auto it = objectToUsdPathMap.find(material);
      it != objectToUsdPathMap.end()) {
    return it->second;
  }

  pxr::SdfPath materialPath =
      allMaterialsPath.AppendChild(pxr::TfToken(material->name().c_str()));
  materialPath = allocateUniquePath(stage, materialPath);

  auto usdMat = pxr::UsdShadeMaterial::Define(stage, materialPath);

  // Create a UsdPreviewSurface shader
  pxr::SdfPath shaderPath =
      materialPath.AppendChild(pxr::TfToken("PreviewSurface"));
  auto previewShader = pxr::UsdShadeShader::Define(stage, shaderPath);
  previewShader.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdPreviewSurface")));

  if (material->subtype() == tokens::material::matte) {
    // Get material parameters with defaults
    float3 color{1.f, 0.f, 0.f};
    float opacity = 1.f;

    auto *colorParam = material->parameter("color");
    auto *colorSampler = material->parameterValueAsObject<Sampler>("color");
    auto *opacityParam = material->parameter("opacity");
    auto *opacitySampler = material->parameterValueAsObject<Sampler>("opacity");

    // Handle color input (texture or value)
    if (colorSampler) {
      auto colorTextureShader =
          tsdSamplertoUSD(stage, materialPath, "diffuseColor", colorSampler);
      auto colorInput = previewShader.CreateInput(
          pxr::TfToken("diffuseColor"), pxr::SdfValueTypeNames->Color3f);
      colorInput.ConnectToSource(
          colorTextureShader.ConnectableAPI(), pxr::TfToken("rgb"));
    } else {
      if (colorParam && colorParam->value().is<float3>()) {
        color = colorParam->value().get<float3>();
      }
      previewShader
          .CreateInput(
              pxr::TfToken("diffuseColor"), pxr::SdfValueTypeNames->Color3f)
          .Set(pxr::GfVec3f(color.x, color.y, color.z));
    }

    // Handle opacity input (texture or value)
    if (opacitySampler) {
      auto opacityTextureShader =
          tsdSamplertoUSD(stage, materialPath, "opacity", opacitySampler);
      auto opacityInput = previewShader.CreateInput(
          pxr::TfToken("opacity"), pxr::SdfValueTypeNames->Float);
      auto opacityChannel = tsdSamplertoScalarOutputToken(opacitySampler);
      opacityInput.ConnectToSource(
          opacityTextureShader.ConnectableAPI(), opacityChannel);
    } else {
      if (opacityParam && opacityParam->value().is<float>()) {
        opacity = opacityParam->value().get<float>();
      }
      previewShader
          .CreateInput(pxr::TfToken("opacity"), pxr::SdfValueTypeNames->Float)
          .Set(opacity);
    }

    // Set material type properties
    previewShader
        .CreateInput(pxr::TfToken("roughness"), pxr::SdfValueTypeNames->Float)
        .Set(1.0f); // Matte materials are fully rough
    previewShader
        .CreateInput(pxr::TfToken("metallic"), pxr::SdfValueTypeNames->Float)
        .Set(0.0f); // Matte materials are non-metallic

  } else if (material->subtype() == tokens::material::physicallyBased) {
    // Get material parameters with defaults
    float3 baseColor{0.8f, 0.8f, 0.8f};
    float opacity = 1.f;
    float metallic = 1.f;
    float roughness = 1.f;
    float3 emissive{0.f, 0.f, 0.f};
    float ior = 1.5f;
    float clearcoat = 0.f;
    float clearcoatRoughness = 0.f;

    // Get parameter objects
    auto *baseColorParam = material->parameter("baseColor");
    auto *baseColorSampler =
        material->parameterValueAsObject<Sampler>("baseColor");
    auto *opacityParam = material->parameter("opacity");
    auto *opacitySampler = material->parameterValueAsObject<Sampler>("opacity");
    auto *metallicParam = material->parameter("metallic");
    auto *metallicSampler =
        material->parameterValueAsObject<Sampler>("metallic");
    auto *roughnessParam = material->parameter("roughness");
    auto *roughnessSampler =
        material->parameterValueAsObject<Sampler>("roughness");
    auto *emissiveParam = material->parameter("emissive");
    auto *emissiveSampler =
        material->parameterValueAsObject<Sampler>("emissive");
    auto *normalSampler = material->parameterValueAsObject<Sampler>("normal");
    auto *iorParam = material->parameter("ior");
    auto *clearcoatParam = material->parameter("clearcoat");
    auto *clearcoatRoughnessParam = material->parameter("clearcoatRoughness");

    // Handle baseColor input (texture or value)
    if (baseColorSampler) {
      auto colorTextureShader = tsdSamplertoUSD(
          stage, materialPath, "diffuseColor", baseColorSampler);
      auto colorInput = previewShader.CreateInput(
          pxr::TfToken("diffuseColor"), pxr::SdfValueTypeNames->Color3f);
      colorInput.ConnectToSource(
          colorTextureShader.ConnectableAPI(), pxr::TfToken("rgb"));
    } else {
      if (baseColorParam && baseColorParam->value().is<float3>()) {
        baseColor = baseColorParam->value().get<float3>();
      }
      previewShader
          .CreateInput(
              pxr::TfToken("diffuseColor"), pxr::SdfValueTypeNames->Color3f)
          .Set(pxr::GfVec3f(baseColor.x, baseColor.y, baseColor.z));
    }

    // Handle opacity input (texture or value)
    if (opacitySampler) {
      auto opacityTextureShader =
          tsdSamplertoUSD(stage, materialPath, "opacity", opacitySampler);
      auto opacityInput = previewShader.CreateInput(
          pxr::TfToken("opacity"), pxr::SdfValueTypeNames->Float);
      auto opacityChannel = tsdSamplertoScalarOutputToken(opacitySampler);
      opacityInput.ConnectToSource(
          opacityTextureShader.ConnectableAPI(), opacityChannel);
    } else {
      if (opacityParam && opacityParam->value().is<float>()) {
        opacity = opacityParam->value().get<float>();
      }
      previewShader
          .CreateInput(pxr::TfToken("opacity"), pxr::SdfValueTypeNames->Float)
          .Set(opacity);
    }

    // Handle metallic input (texture or value)
    if (metallicSampler) {
      auto metallicTextureShader =
          tsdSamplertoUSD(stage, materialPath, "metallic", metallicSampler);
      auto metallicInput = previewShader.CreateInput(
          pxr::TfToken("metallic"), pxr::SdfValueTypeNames->Float);
      auto metallicChannel = tsdSamplertoScalarOutputToken(metallicSampler);
      metallicInput.ConnectToSource(
          metallicTextureShader.ConnectableAPI(), metallicChannel);
    } else {
      if (metallicParam && metallicParam->value().is<float>()) {
        metallic = metallicParam->value().get<float>();
      }
      previewShader
          .CreateInput(pxr::TfToken("metallic"), pxr::SdfValueTypeNames->Float)
          .Set(metallic);
    }

    // Handle roughness input (texture or value)
    if (roughnessSampler) {
      auto roughnessTextureShader =
          tsdSamplertoUSD(stage, materialPath, "roughness", roughnessSampler);
      auto roughnessInput = previewShader.CreateInput(
          pxr::TfToken("roughness"), pxr::SdfValueTypeNames->Float);
      auto roughnessChannel = tsdSamplertoScalarOutputToken(roughnessSampler);
      roughnessInput.ConnectToSource(
          roughnessTextureShader.ConnectableAPI(), roughnessChannel);
    } else {
      if (roughnessParam && roughnessParam->value().is<float>()) {
        roughness = roughnessParam->value().get<float>();
      }
      previewShader
          .CreateInput(pxr::TfToken("roughness"), pxr::SdfValueTypeNames->Float)
          .Set(roughness);
    }

    // Handle emissive input (texture or value)
    if (emissiveSampler) {
      auto emissiveTextureShader =
          tsdSamplertoUSD(stage, materialPath, "emissive", emissiveSampler);
      auto emissiveInput = previewShader.CreateInput(
          pxr::TfToken("emissiveColor"), pxr::SdfValueTypeNames->Color3f);
      emissiveInput.ConnectToSource(
          emissiveTextureShader.ConnectableAPI(), pxr::TfToken("rgb"));
    } else {
      if (emissiveParam && emissiveParam->value().is<float3>()) {
        emissive = emissiveParam->value().get<float3>();
      }
      previewShader
          .CreateInput(
              pxr::TfToken("emissiveColor"), pxr::SdfValueTypeNames->Color3f)
          .Set(pxr::GfVec3f(emissive.x, emissive.y, emissive.z));
    }

    // Handle normal map
    if (normalSampler) {
      auto normalTextureShader =
          tsdSamplertoUSD(stage, materialPath, "normal", normalSampler, true);
      auto normalInput = previewShader.CreateInput(
          pxr::TfToken("normal"), pxr::SdfValueTypeNames->Normal3f);
      normalInput.ConnectToSource(
          normalTextureShader.ConnectableAPI(), pxr::TfToken("rgb"));
    }

    // Handle scalar parameters (no texture support in UsdPreviewSurface for
    // these)
    if (iorParam && iorParam->value().is<float>()) {
      ior = iorParam->value().get<float>();
    }
    previewShader
        .CreateInput(pxr::TfToken("ior"), pxr::SdfValueTypeNames->Float)
        .Set(ior);

    if (clearcoatParam && clearcoatParam->value().is<float>()) {
      clearcoat = clearcoatParam->value().get<float>();
    }
    previewShader
        .CreateInput(pxr::TfToken("clearcoat"), pxr::SdfValueTypeNames->Float)
        .Set(clearcoat);

    if (clearcoatRoughnessParam
        && clearcoatRoughnessParam->value().is<float>()) {
      clearcoatRoughness = clearcoatRoughnessParam->value().get<float>();
    }
    previewShader
        .CreateInput(
            pxr::TfToken("clearcoatRoughness"), pxr::SdfValueTypeNames->Float)
        .Set(clearcoatRoughness);
  }

  // Create material outputs and connect to shader
  auto materialOutput = usdMat.CreateSurfaceOutput();
  materialOutput.ConnectToSource(
      previewShader.ConnectableAPI(), pxr::TfToken("surface"));

  objectToUsdPathMap.insert({material, materialPath});
  return materialPath;
}

static pxr::SdfPath tsdSurfaceToUSD(
    pxr::UsdStageRefPtr &stage, const Surface *surface)
{
  if (auto it = objectToUsdPathMap.find(surface);
      it != objectToUsdPathMap.end()) {
    return it->second;
  }

  auto *anariGeom =
      surface->parameterValueAsObject<Geometry>(tokens::surface::geometry);

  if (!anariGeom) {
    return pxr::SdfPath::EmptyPath();
  }

  // First export the material if any
  auto *anariMaterial =
      surface->parameterValueAsObject<Material>(tokens::surface::material);

  pxr::SdfPath materialPath;
  if (anariMaterial) {
    materialPath = tsdMaterialToPreviewSurfaceUSD(stage, anariMaterial);
  }

  pxr::SdfPath surfacePath =
      allSurfacesPath.AppendChild(pxr::TfToken(surface->name().c_str()));
  surfacePath = allocateUniquePath(stage, surfacePath);

  if (anariGeom->subtype() == tokens::geometry::triangle) {
    auto usdGeomMesh = pxr::UsdGeomMesh::Define(stage, surfacePath);

    auto triangles =
        anariGeom->parameterValueAsObject<Array>("vertex.position");
    auto points = triangles->dataAs<pxr::GfVec3f>();
    auto pointsCount = triangles->size();
    usdGeomMesh.CreatePointsAttr().Set(
        pxr::VtArray<pxr::GfVec3f>(points, points + pointsCount));

    std::vector<int> faceVertexCounts;
    std::vector<int> faceVertexIndices;

    if (auto triangleIds =
            anariGeom->parameterValueAsObject<Array>("primitive.index")) {
      auto triangles = triangleIds->dataAs<pxr::GfVec3i>();
      auto triangleCount = triangleIds->size();

      faceVertexCounts.resize(triangleCount, 3);
      faceVertexIndices.resize(triangleCount * 3);
      std::copy_n(
          (int *)triangles, triangleCount * 3, faceVertexIndices.begin());
    } else {
      faceVertexCounts.resize(pointsCount / 3, 3);
      faceVertexIndices.resize(pointsCount);
      std::iota(begin(faceVertexIndices), end(faceVertexIndices), 0);
    }

    usdGeomMesh.CreateFaceVertexCountsAttr().Set(
        pxr::VtArray<int>(cbegin(faceVertexCounts), cend(faceVertexCounts)));
    usdGeomMesh.CreateFaceVertexIndicesAttr().Set(
        pxr::VtArray<int>(cbegin(faceVertexIndices), cend(faceVertexIndices)));

    // Export vertex attributes as primvars with proper precedence
    // Precedence order: faceVarying > vertex > primitive (only export the
    // highest precedence one)
    const char *faceVaryingAttributeNames[] = {"faceVarying.attribute0",
        "faceVarying.attribute1",
        "faceVarying.attribute2",
        "faceVarying.attribute3",
        "color"};
    const char *vertexAttributeNames[] = {"vertex.attribute0",
        "vertex.attribute1",
        "vertex.attribute2",
        "vertex.attribute3",
        "color"};
    const char *primitiveAttributeNames[] = {"primitive.attribute0",
        "primitive.attribute1",
        "primitive.attribute2",
        "primitive.attribute3",
        "color"};
    const char *primvarNames[] = {
        "attribute0", "attribute1", "attribute2", "attribute3", "displayColor"};

    auto primvarsAPI = pxr::UsdGeomPrimvarsAPI(usdGeomMesh);

    for (int i = 0; i < 4; ++i) {
      // Check in precedence order: faceVarying > vertex > primitive
      auto array = anariGeom->parameterValueAsObject<Array>(
          faceVaryingAttributeNames[i]);
      auto interpolation = pxr::UsdGeomTokens->faceVarying;
      if (!array) {
        array =
            anariGeom->parameterValueAsObject<Array>(vertexAttributeNames[i]);
        interpolation = pxr::UsdGeomTokens->vertex;
      }
      if (!array) {
        array = anariGeom->parameterValueAsObject<Array>(
            primitiveAttributeNames[i]);
        interpolation = pxr::UsdGeomTokens->uniform;
      }

      if (array) {
        auto elementType = array->elementType();
        auto elementCount = array->size();

        switch (elementType) {
        case ANARI_FLOAT32: {
          auto primvar =
              primvarsAPI.CreatePrimvar(pxr::TfToken(primvarNames[i]),
                  pxr::SdfValueTypeNames->FloatArray,
                  interpolation);
          auto data = array->dataAs<float>();
          primvar.Set(pxr::VtArray<float>(data, data + elementCount));
          break;
        }
        case ANARI_FLOAT32_VEC2: {
          auto primvar =
              primvarsAPI.CreatePrimvar(pxr::TfToken(primvarNames[i]),
                  pxr::SdfValueTypeNames->Float2Array,
                  interpolation);
          auto data = array->dataAs<pxr::GfVec2f>();
          primvar.Set(pxr::VtArray<pxr::GfVec2f>(data, data + elementCount));
          break;
        }
        case ANARI_FLOAT32_VEC3: {
          auto primvar =
              primvarsAPI.CreatePrimvar(pxr::TfToken(primvarNames[i]),
                  pxr::SdfValueTypeNames->Float3Array,
                  interpolation);
          auto data = array->dataAs<pxr::GfVec3f>();
          primvar.Set(pxr::VtArray<pxr::GfVec3f>(data, data + elementCount));
          break;
        }
        case ANARI_FLOAT32_VEC4: {
          auto primvar =
              primvarsAPI.CreatePrimvar(pxr::TfToken(primvarNames[i]),
                  pxr::SdfValueTypeNames->Float4Array,
                  interpolation);
          auto data = array->dataAs<pxr::GfVec4f>();
          primvar.Set(pxr::VtArray<pxr::GfVec4f>(data, data + elementCount));
          break;
        }
        case ANARI_UINT8_VEC3: {
          auto primvar =
              primvarsAPI.CreatePrimvar(pxr::TfToken(primvarNames[i]),
                  pxr::SdfValueTypeNames->Color3fArray,
                  interpolation);
          auto srcData = array->dataAs<uint8_t>();
          pxr::VtArray<pxr::GfVec3f> colorData(elementCount);
          for (size_t j = 0; j < elementCount; ++j) {
            colorData[j] = pxr::GfVec3f(srcData[j * 3 + 0] / 255.0f,
                srcData[j * 3 + 1] / 255.0f,
                srcData[j * 3 + 2] / 255.0f);
          }
          primvar.Set(colorData);
          break;
        }
        case ANARI_UINT8_VEC4: {
          auto primvar =
              primvarsAPI.CreatePrimvar(pxr::TfToken(primvarNames[i]),
                  pxr::SdfValueTypeNames->Color4fArray,
                  interpolation);
          auto srcData = array->dataAs<uint8_t>();
          pxr::VtArray<pxr::GfVec4f> colorData(elementCount);
          for (size_t j = 0; j < elementCount; ++j) {
            colorData[j] = pxr::GfVec4f(srcData[j * 4 + 0] / 255.0f,
                srcData[j * 4 + 1] / 255.0f,
                srcData[j * 4 + 2] / 255.0f,
                srcData[j * 4 + 3] / 255.0f);
          }
          primvar.Set(colorData);
          break;
        }
        }
      }
    }

    // Bind material to the mesh if available
    if (!materialPath.IsEmpty()) {
      auto materialBinding =
          pxr::UsdShadeMaterialBindingAPI::Apply(usdGeomMesh.GetPrim());
      auto usdMaterial = pxr::UsdShadeMaterial::Get(stage, materialPath);
      materialBinding.Bind(usdMaterial);
    }
  }

  objectToUsdPathMap.insert({surface, surfacePath});

  return surfacePath;
}

void export_SceneToUSD(Scene &scene, const char *filename, int framesPerSecond)
{
  // Clear some global state (!!!)
  usedTextureFileNames.clear();
  samplerToFilepathMap.clear();
  objectToUsdPathMap.clear();

  auto outputFilename = std::filesystem::path(filename);
  outputPath = outputFilename.parent_path() / outputFilename.stem();

  tsd::core::logStatus("Exporting scene to USD file: %s", filename);

  pxr::UsdStageRefPtr stage = pxr::UsdStage::CreateNew(filename);
  if (!stage) {
    tsd::core::logError("Failed to create USD stage for file: %s", filename);
    return;
  }

  const float originalTime = scene.getAnimationTime();
  const int exportFps = std::max(1, framesPerSecond);

  pxr::SdfPath currentPath = allLayersPath;
  std::unordered_set<pxr::SdfPath, pxr::SdfPath::Hash> existingNames;

  for (auto l : scene.layers()) {
    if (auto layer = l.second.ptr) {
      int currentLevel = -1;
      std::stack<math::mat4> transformStack;

      layer->traverse(layer->root(), [&](const LayerNode &node, int level) {
        // Handle depth traversal
        if (level == 0) {
          // special case for root -- output UsdGeomScope
          currentPath = currentPath.AppendChild(pxr::TfToken(l.first.c_str()));
          pxr::UsdGeomScope::Define(stage, currentPath);
          transformStack.push(math::mat4(math::identity));

          return true;
        }

        if (level > currentLevel) {
          for (int i = currentLevel; i < level; i++)
            transformStack.push(transformStack.top());

        } else if (level < currentLevel) {
          for (int i = level; i < currentLevel; i++)
            transformStack.pop();
        }

        currentLevel = level;

        // Handle this node
        if (node->isEmpty())
          return true;
        if (node->isTransform()) {
          auto transform = node->getTransform();
          transformStack.pop();
          transformStack.push(math::mul(transformStack.top(), transform));
        }
        if (node->isObject()) {
          auto object = node->getObject();
          auto name = node->name();
          if (name.empty())
            name = object->name();
          if (name.empty()) {
            fprintf(stderr, "Don't know what to do yet");
            std::exit(1);
          }

          auto objectPath = currentPath.AppendChild(pxr::TfToken(name.c_str()));
          objectPath = allocateUniquePath(
              stage, currentPath.AppendChild(pxr::TfToken(name.c_str())));

          switch (object->type()) {
          case ANARI_CAMERA: {
            auto camera = static_cast<const Camera *>(object);
            auto usdCamera = pxr::UsdGeomCamera::Define(stage, objectPath);

            size_t cameraSampleCount = 0;
            for (size_t i = 0; i < scene.numberOfAnimations(); ++i) {
              if (auto *anim = scene.animation(i);
                  anim && anim->targetsObject(camera)) {
                cameraSampleCount = anim->timeStepCount();
                break;
              }
            }

            if (camera->subtype() == tokens::camera::orthographic) {
              usdCamera.GetProjectionAttr().Set(
                  pxr::UsdGeomTokens->orthographic);
            } else {
              usdCamera.GetProjectionAttr().Set(
                  pxr::UsdGeomTokens->perspective);
            }

            auto xformOp =
                usdCamera.AddXformOp(pxr::UsdGeomXformOp::TypeTransform);

            auto setCameraSample = [&](double timeCode, bool timeSampled) {
              const auto position =
                  camera->parameterValueAs<math::float3>("position")
                      .value_or(math::float3(0.f, 0.f, 0.f));
              const auto direction =
                  camera->parameterValueAs<math::float3>("direction")
                      .value_or(math::float3(0.f, 0.f, -1.f));
              const auto up =
                  camera->parameterValueAs<math::float3>("up").value_or(
                      math::float3(0.f, 1.f, 0.f));

              const auto at = position + direction;
              const auto view = linalg::lookat_matrix(position, at, up);
              const auto world = linalg::inverse(view);
              const auto xfm = math::mul(transformStack.top(), world);

              const auto usdXfm = pxr::GfMatrix4d(xfm[0][0],
                  xfm[0][1],
                  xfm[0][2],
                  xfm[0][3],
                  xfm[1][0],
                  xfm[1][1],
                  xfm[1][2],
                  xfm[1][3],
                  xfm[2][0],
                  xfm[2][1],
                  xfm[2][2],
                  xfm[2][3],
                  xfm[3][0],
                  xfm[3][1],
                  xfm[3][2],
                  xfm[3][3]);

              if (timeSampled)
                xformOp.Set(usdXfm, timeCode);
              else
                xformOp.Set(usdXfm);

              if (camera->subtype() == tokens::camera::orthographic) {
                const auto height =
                    camera->parameterValueAs<float>("height").value_or(1.f);
                if (timeSampled) {
                  usdCamera.GetVerticalApertureAttr().Set(height, timeCode);
                  usdCamera.GetHorizontalApertureAttr().Set(height, timeCode);
                } else {
                  usdCamera.GetVerticalApertureAttr().Set(height);
                  usdCamera.GetHorizontalApertureAttr().Set(height);
                }
              } else {
                const auto fovy =
                    camera->parameterValueAs<float>("fovy").value_or(
                        float(M_PI) / 3.0f);
                constexpr float verticalAperture = 24.0f;
                const float focalLength =
                    0.5f * verticalAperture / std::tan(0.5f * fovy);
                if (timeSampled) {
                  usdCamera.GetVerticalApertureAttr().Set(
                      verticalAperture, timeCode);
                  usdCamera.GetHorizontalApertureAttr().Set(
                      verticalAperture, timeCode);
                  usdCamera.GetFocalLengthAttr().Set(focalLength, timeCode);
                } else {
                  usdCamera.GetVerticalApertureAttr().Set(verticalAperture);
                  usdCamera.GetHorizontalApertureAttr().Set(verticalAperture);
                  usdCamera.GetFocalLengthAttr().Set(focalLength);
                }
              }

              auto nearVal = camera->parameterValueAs<float>("near");
              auto farVal = camera->parameterValueAs<float>("far");
              if (nearVal && farVal && *farVal > *nearVal && *farVal > 0.f) {
                const auto clip = pxr::GfVec2f(*nearVal, *farVal);
                if (timeSampled)
                  usdCamera.GetClippingRangeAttr().Set(clip, timeCode);
                else
                  usdCamera.GetClippingRangeAttr().Set(clip);
              }
            };

            if (cameraSampleCount > 1) {
              const double timeCodesPerSecond = static_cast<double>(exportFps);
              stage->SetStartTimeCode(0.0);
              stage->SetEndTimeCode(static_cast<double>(cameraSampleCount - 1));
              stage->SetTimeCodesPerSecond(timeCodesPerSecond);
              stage->SetFramesPerSecond(timeCodesPerSecond);

              for (size_t i = 0; i < cameraSampleCount; ++i) {
                const double tNorm = static_cast<double>(i)
                    / static_cast<double>(cameraSampleCount - 1);
                scene.setAnimationTime(static_cast<float>(tNorm));
                setCameraSample(static_cast<double>(i), true);
              }
              scene.setAnimationTime(originalTime);
            } else {
              setCameraSample(0.0, false);
            }

            break;
          }
          case ANARI_SURFACE: {
            if (auto meshPath = tsdSurfaceToUSD(
                    stage, static_cast<const Surface *>(object));
                meshPath != pxr::SdfPath::EmptyPath()) {
              tsd::core::logInfo("Exporting surface %s (%p) to %s",
                  name.c_str(),
                  object,
                  meshPath.GetText());
              auto mesh = pxr::UsdGeomMesh::Define(stage, objectPath);
              mesh.GetPrim().GetReferences().AddInternalReference(meshPath);
              auto xfm = transformStack.top();
              mesh.AddXformOp(pxr::UsdGeomXformOp::TypeTransform)
                  .Set(pxr::GfMatrix4d(xfm[0][0],
                      xfm[0][1],
                      xfm[0][2],
                      xfm[0][3],
                      xfm[1][0],
                      xfm[1][1],
                      xfm[1][2],
                      xfm[1][3],
                      xfm[2][0],
                      xfm[2][1],
                      xfm[2][2],
                      xfm[2][3],
                      xfm[3][0],
                      xfm[3][1],
                      xfm[3][2],
                      xfm[3][3]));
            }
            break;
          }
          default:
            break;
          }
        }
        return true;
      });
    }
  }

  stage->GetPrimAtPath(allObjectsPath)
      .GetPrimStack()[0]
      ->SetSpecifier(pxr::SdfSpecifierOver);

  stage->Save();

  constexpr const auto timeout = 100ms;
  while (!textureWriteQueue.empty()) {
    for (auto &fut : textureWriteQueue) {
      if (fut.valid() && fut.wait_for(timeout) == std::future_status::ready) {
        tsd::core::logInfo(fut.get().c_str());
        swap(fut, textureWriteQueue.back());
      }
    }
    textureWriteQueue.erase(
        std::remove_if(begin(textureWriteQueue),
            end(textureWriteQueue),
            [](const std::future<std::string> &fut) { return !fut.valid(); }),
        end(textureWriteQueue));
  }

  tsd::core::logStatus("...done exporting USD scene to file: %s", filename);
}
#else

namespace tsd::io {
void export_SceneToUSD(Scene &, const char *, int)
{
  tsd::core::logError("[export_USD] USD not enabled in TSD build.");
}
#endif

} // namespace tsd::io
