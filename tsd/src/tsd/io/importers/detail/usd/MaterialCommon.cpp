// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#if TSD_USE_USD

#include "MaterialCommon.h"

// tsd_core
#include <tsd/core/Logging.hpp>

// pxr
#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdShade/input.h>

namespace tsd::io::materials {

bool getShaderFloatInput(
    const pxr::UsdShadeShader &shader, const char *inputName, float &outValue)
{
  pxr::UsdShadeInput input = shader.GetInput(pxr::TfToken(inputName));
  if (!input) {
    return false;
  }

  // Check if there's a connected source
  if (input.HasConnectedSource()) {
    pxr::UsdShadeConnectableAPI source;
    pxr::TfToken sourceName;
    pxr::UsdShadeAttributeType sourceType;
    if (input.GetConnectedSource(&source, &sourceName, &sourceType)) {
      // Check if this is a connection to a material interface input
      pxr::UsdPrim sourcePrim = source.GetPrim();
      if (sourcePrim.IsA<pxr::UsdShadeMaterial>()) {
        // This is a material interface connection - get the value from the
        // material's input
        pxr::UsdShadeMaterial mat(sourcePrim);
        pxr::UsdShadeInput matInput = mat.GetInput(sourceName);
        if (matInput && matInput.Get(&outValue)) {
          return true;
        }
      } else {
        // This is a connection to another shader's output
        pxr::UsdShadeShader sourceShader(sourcePrim);
        if (sourceShader) {
          pxr::UsdShadeOutput output = sourceShader.GetOutput(sourceName);
          if (output) {
            pxr::UsdAttribute attr = output.GetAttr();
            if (attr && attr.Get(&outValue)) {
              return true;
            }
          }
        }
      }
    }
  }

  // Fall back to direct value
  if (input.Get(&outValue)) {
    return true;
  }
  return false;
}

bool getShaderBoolInput(
    const pxr::UsdShadeShader &shader, const char *inputName, bool &outValue)
{
  pxr::UsdShadeInput input = shader.GetInput(pxr::TfToken(inputName));
  if (!input) {
    return false;
  }

  // Check if there's a connected source
  if (input.HasConnectedSource()) {
    pxr::UsdShadeConnectableAPI source;
    pxr::TfToken sourceName;
    pxr::UsdShadeAttributeType sourceType;
    if (input.GetConnectedSource(&source, &sourceName, &sourceType)) {
      // Check if this is a connection to a material interface input
      pxr::UsdPrim sourcePrim = source.GetPrim();
      if (sourcePrim.IsA<pxr::UsdShadeMaterial>()) {
        // This is a material interface connection - get the value from the
        // material's input
        pxr::UsdShadeMaterial mat(sourcePrim);
        pxr::UsdShadeInput matInput = mat.GetInput(sourceName);
        if (matInput && matInput.Get(&outValue)) {
          return true;
        }
      } else {
        // This is a connection to another shader's output
        pxr::UsdShadeShader sourceShader(sourcePrim);
        if (sourceShader) {
          pxr::UsdShadeOutput output = sourceShader.GetOutput(sourceName);
          if (output) {
            pxr::UsdAttribute attr = output.GetAttr();
            if (attr && attr.Get(&outValue)) {
              return true;
            }
          }
        }
      }
    }
  }

  // Fall back to direct value
  if (input.Get(&outValue)) {
    return true;
  }
  return false;
}

bool getShaderColorInput(const pxr::UsdShadeShader &shader,
    const char *inputName,
    pxr::GfVec3f &outValue)
{
  pxr::UsdShadeInput input = shader.GetInput(pxr::TfToken(inputName));
  if (!input) {
    return false;
  }

  // Check if there's a connected source
  if (input.HasConnectedSource()) {
    pxr::UsdShadeConnectableAPI source;
    pxr::TfToken sourceName;
    pxr::UsdShadeAttributeType sourceType;
    if (input.GetConnectedSource(&source, &sourceName, &sourceType)) {
      // Check if this is a connection to a material interface input
      pxr::UsdPrim sourcePrim = source.GetPrim();
      if (sourcePrim.IsA<pxr::UsdShadeMaterial>()) {
        // This is a material interface connection - get the value from the
        // material's input
        pxr::UsdShadeMaterial mat(sourcePrim);
        pxr::UsdShadeInput matInput = mat.GetInput(sourceName);
        if (matInput && matInput.Get(&outValue)) {
          return true;
        }
      } else {
        // This is a connection to another shader's output
        pxr::UsdShadeShader sourceShader(sourcePrim);
        if (sourceShader) {
          pxr::UsdShadeOutput output = sourceShader.GetOutput(sourceName);
          if (output) {
            pxr::UsdAttribute attr = output.GetAttr();
            if (attr && attr.Get(&outValue)) {
              return true;
            }
          }
        }
      }
    }
  }

  // Fall back to direct value
  if (input.Get(&outValue)) {
    return true;
  }
  return false;
}

bool getShaderTextureInput(const pxr::UsdShadeShader &shader,
    const char *inputName,
    std::string &outFilePath)
{
  pxr::UsdShadeInput input = shader.GetInput(pxr::TfToken(inputName));
  if (!input) {
    return false;
  }

  // Check if there's a connected texture reader
  if (input.HasConnectedSource()) {
    pxr::UsdShadeConnectableAPI source;
    pxr::TfToken sourceName;
    pxr::UsdShadeAttributeType sourceType;
    input.GetConnectedSource(&source, &sourceName, &sourceType);

    // Check if this is a connection to a material interface input
    pxr::UsdPrim sourcePrim = source.GetPrim();
    if (sourcePrim.IsA<pxr::UsdShadeMaterial>()) {
      // This is a material interface connection - get the value from the
      // material's input
      pxr::UsdShadeMaterial mat(sourcePrim);
      pxr::UsdShadeInput matInput = mat.GetInput(sourceName);
      if (matInput) {
        pxr::SdfAssetPath assetPath;
        if (matInput.Get(&assetPath)) {
          outFilePath = assetPath.GetResolvedPath();
          if (outFilePath.empty()) {
            outFilePath = assetPath.GetAssetPath();
          }
          return !outFilePath.empty();
        }
      }
    } else {
      // Check if this is a texture reader shader
      pxr::UsdShadeShader textureShader(sourcePrim);
      if (textureShader) {
        // Look for file input on the texture reader
        pxr::UsdShadeInput fileInput =
            textureShader.GetInput(pxr::TfToken("file"));
        if (fileInput) {
          pxr::SdfAssetPath assetPath;
          if (fileInput.Get(&assetPath)) {
            outFilePath = assetPath.GetResolvedPath();
            if (outFilePath.empty()) {
              outFilePath = assetPath.GetAssetPath();
            }
            return !outFilePath.empty();
          }
        }
      }
    }
  }

  // Try direct asset path input
  pxr::SdfAssetPath assetPath;
  if (input.Get(&assetPath)) {
    outFilePath = assetPath.GetResolvedPath();
    if (outFilePath.empty()) {
      outFilePath = assetPath.GetAssetPath();
    }
    return !outFilePath.empty();
  }

  return false;
}

// Walk input:st of a UsdUVTexture through optional UsdTransform2d and
// UsdPrimvarReader_float2 to extract inAttribute and inTransform.
static void resolveSTChain(const pxr::UsdShadeShader &texShader,
    std::string &outAttribute,
    bool &outHasTransform,
    tsd::math::mat4 &outTransform)
{
  outAttribute = "attribute0";
  outHasTransform = false;

  pxr::UsdShadeInput stInput = texShader.GetInput(pxr::TfToken("st"));
  if (!stInput || !stInput.HasConnectedSource())
    return;

  pxr::UsdShadeConnectableAPI stSource;
  pxr::TfToken stSourceName;
  pxr::UsdShadeAttributeType stSourceType;
  if (!stInput.GetConnectedSource(&stSource, &stSourceName, &stSourceType))
    return;

  pxr::UsdShadeShader stShader(stSource.GetPrim());
  if (!stShader)
    return;

  pxr::TfToken stShaderId;
  stShader.GetShaderId(&stShaderId);

  // Case 1: directly connected to UsdPrimvarReader_float2
  if (stShaderId == pxr::TfToken("UsdPrimvarReader_float2")) {
    pxr::UsdShadeInput varnameInput =
        stShader.GetInput(pxr::TfToken("varname"));
    if (varnameInput) {
      std::string varname;
      if (varnameInput.Get(&varname) && !varname.empty())
        outAttribute = varname;
    }
    return;
  }

  // Case 2: connected to UsdTransform2d
  if (stShaderId == pxr::TfToken("UsdTransform2d")) {
    pxr::GfVec2f scale(1.0f, 1.0f);
    pxr::GfVec2f translation(0.0f, 0.0f);

    if (auto scaleIn = stShader.GetInput(pxr::TfToken("scale")))
      scaleIn.Get(&scale);
    if (auto transIn = stShader.GetInput(pxr::TfToken("translation")))
      transIn.Get(&translation);

    outTransform = tsd::math::IDENTITY_MAT4;
    outTransform[0][0] = scale[0];
    outTransform[1][1] = scale[1];
    outTransform[3][0] = translation[0];
    outTransform[3][1] = translation[1];
    outHasTransform = true;

    // Follow inputs:in to UsdPrimvarReader_float2
    pxr::UsdShadeInput inInput = stShader.GetInput(pxr::TfToken("in"));
    if (inInput && inInput.HasConnectedSource()) {
      pxr::UsdShadeConnectableAPI inSource;
      pxr::TfToken inSourceName;
      pxr::UsdShadeAttributeType inSourceType;
      if (inInput.GetConnectedSource(
              &inSource, &inSourceName, &inSourceType)) {
        pxr::UsdShadeShader readerShader(inSource.GetPrim());
        if (readerShader) {
          pxr::TfToken readerId;
          readerShader.GetShaderId(&readerId);
          if (readerId == pxr::TfToken("UsdPrimvarReader_float2")) {
            pxr::UsdShadeInput varnameInput =
                readerShader.GetInput(pxr::TfToken("varname"));
            if (varnameInput) {
              std::string varname;
              if (varnameInput.Get(&varname) && !varname.empty())
                outAttribute = varname;
            }
          }
        }
      }
    }
  }
}

SamplerRef resolveTexturedInput(Scene &scene,
    const pxr::UsdShadeShader &shader,
    const char *inputName,
    const std::string &basePath,
    TextureCache &texCache)
{
  pxr::UsdShadeInput input = shader.GetInput(pxr::TfToken(inputName));
  if (!input || !input.HasConnectedSource())
    return {};

  pxr::UsdShadeConnectableAPI source;
  pxr::TfToken sourceName;
  pxr::UsdShadeAttributeType sourceType;
  if (!input.GetConnectedSource(&source, &sourceName, &sourceType))
    return {};

  pxr::UsdShadeShader texShader(source.GetPrim());
  if (!texShader)
    return {};

  pxr::TfToken shaderId;
  texShader.GetShaderId(&shaderId);
  if (shaderId != pxr::TfToken("UsdUVTexture"))
    return {};

  // Load the texture file
  pxr::UsdShadeInput fileInput = texShader.GetInput(pxr::TfToken("file"));
  if (!fileInput)
    return {};

  pxr::SdfAssetPath assetPath;
  if (!fileInput.Get(&assetPath))
    return {};

  std::string filePath = assetPath.GetResolvedPath();
  if (filePath.empty())
    filePath = assetPath.GetAssetPath();
  if (filePath.empty())
    return {};

  if (filePath[0] != '/')
    filePath = basePath + filePath;

  auto sampler = importTexture(scene, filePath, texCache, false);
  if (!sampler) {
    logWarning("[import_USD] Failed to load texture for input '%s': %s\n",
        inputName,
        filePath.c_str());
    return {};
  }

  // Read wrap modes from UsdUVTexture
  auto readWrapMode = [&](const char *usdName) -> std::string {
    pxr::UsdShadeInput wrapInput = texShader.GetInput(pxr::TfToken(usdName));
    if (!wrapInput)
      return "repeat";
    pxr::TfToken wrapToken;
    if (wrapInput.Get(&wrapToken)) {
      std::string w = wrapToken.GetString();
      if (w == "clamp")
        return "clampToEdge";
      if (w == "mirror")
        return "mirrorRepeat";
      return w;
    }
    return "repeat";
  };

  sampler->setParameter("wrapMode1", readWrapMode("wrapS").c_str());
  sampler->setParameter("wrapMode2", readWrapMode("wrapT").c_str());

  // Walk the ST chain for primvar name and optional transform
  std::string attribute;
  bool hasTransform = false;
  tsd::math::mat4 transform = tsd::math::IDENTITY_MAT4;
  resolveSTChain(texShader, attribute, hasTransform, transform);

  sampler->setParameter("inAttribute", attribute.c_str());
  if (hasTransform)
    sampler->setParameter("inTransform", transform);

  logStatus("[import_USD] Resolved textured input '%s': file=%s, attr=%s\n",
      inputName,
      filePath.c_str(),
      attribute.c_str());

  return sampler;
}

} // namespace tsd::io::materials

#endif
