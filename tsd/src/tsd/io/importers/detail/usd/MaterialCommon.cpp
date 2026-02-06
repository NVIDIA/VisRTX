// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#if TSD_USE_USD

#include "MaterialCommon.h"

// pxr
#include <pxr/usd/usdShade/input.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/assetPath.h>

namespace tsd::io::materials {

bool getShaderFloatInput(const pxr::UsdShadeShader &shader,
    const char *inputName,
    float &outValue)
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
        // This is a material interface connection - get the value from the material's input
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

bool getShaderBoolInput(const pxr::UsdShadeShader &shader,
    const char *inputName,
    bool &outValue)
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
        // This is a material interface connection - get the value from the material's input
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
        // This is a material interface connection - get the value from the material's input
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
      // This is a material interface connection - get the value from the material's input
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
        pxr::UsdShadeInput fileInput = textureShader.GetInput(pxr::TfToken("file"));
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

} // namespace tsd::io::materials

#endif
