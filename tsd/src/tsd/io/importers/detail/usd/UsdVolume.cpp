// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdVolume.h"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/io/animation/SpatialFieldFileBinding.hpp"
#include "tsd/io/importers.hpp"
// usd
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/vt/array.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
// std
#include <optional>
#include <string>
#include <vector>

namespace tsd::io::usd {

using namespace tsd::core;

namespace {

// Transfer functions authored on a Stage ////////////////////////////////////

struct VolumeTransferFunction
{
  std::vector<math::float4> colors;
  std::vector<float> xPointsColor;
  std::vector<float> xPoints;
  std::vector<float> opacityValues;
  math::float2 domain{0.0f, 1.0f};
  float unitDistance{0.0f};
  bool hasTransferFunction{false};
};

bool extractColormapFromPrim(
    const pxr::UsdPrim &prim, VolumeTransferFunction &tf)
{
  auto rgbaAttr = prim.GetAttribute(pxr::TfToken("rgbaPoints"));
  if (!rgbaAttr)
    return false;

  pxr::VtArray<pxr::GfVec4f> rgbaPoints;
  if (!rgbaAttr.Get(&rgbaPoints) || rgbaPoints.empty())
    return false;

  tf.colors.resize(rgbaPoints.size());
  for (size_t i = 0; i < rgbaPoints.size(); ++i) {
    const auto &c = rgbaPoints[i];
    tf.colors[i] = math::float4(c[0], c[1], c[2], c[3]);
  }

  auto readFloatArray = [&](const char *name, std::vector<float> &out) {
    if (auto attr = prim.GetAttribute(pxr::TfToken(name))) {
      pxr::VtArray<float> values;
      if (attr.Get(&values))
        out.assign(values.begin(), values.end());
    }
  };

  readFloatArray("xPointsColor", tf.xPointsColor);
  readFloatArray("xPoints", tf.xPoints);
  readFloatArray("opacityValues", tf.opacityValues);

  if (auto attr = prim.GetAttribute(pxr::TfToken("domain"))) {
    pxr::GfVec2f domain;
    if (attr.Get(&domain))
      tf.domain = math::float2(domain[0], domain[1]);
  }

  if (auto attr = prim.GetAttribute(pxr::TfToken("unitDistance"))) {
    float unitDistance = 0.f;
    if (attr.Get(&unitDistance) && unitDistance > 0.0f)
      tf.unitDistance = unitDistance;
  }

  tf.hasTransferFunction = true;
  return true;
}

core::TransferFunction toTransferFunction(const VolumeTransferFunction &vtf)
{
  core::TransferFunction tf;
  tf.range = {vtf.domain.x, vtf.domain.y};

  const auto &xColor =
      vtf.xPointsColor.empty() ? vtf.xPoints : vtf.xPointsColor;
  for (size_t i = 0; i < vtf.colors.size() && i < xColor.size(); ++i) {
    tf.colorPoints.emplace_back(
        xColor[i], vtf.colors[i].x, vtf.colors[i].y, vtf.colors[i].z);
  }

  if (!vtf.opacityValues.empty()) {
    for (size_t i = 0; i < vtf.opacityValues.size() && i < vtf.xPoints.size();
         ++i)
      tf.opacityPoints.emplace_back(vtf.xPoints[i], vtf.opacityValues[i]);
  } else {
    for (size_t i = 0; i < vtf.colors.size() && i < vtf.xPoints.size(); ++i)
      tf.opacityPoints.emplace_back(vtf.xPoints[i], vtf.colors[i].w);
  }

  return tf;
}

VolumeTransferFunction getVolumeTransferFunction(const pxr::UsdPrim &prim)
{
  VolumeTransferFunction tf;

  // Material binding chain: Material -> VolumeShader -> Colormap.
  if (pxr::UsdShadeMaterialBindingAPI::CanApply(prim)) {
    pxr::UsdShadeMaterialBindingAPI binding(prim);
    pxr::UsdShadeMaterial usdMaterial;

    if (auto materialRel =
            prim.GetRelationship(pxr::TfToken("material:binding"))) {
      pxr::SdfPathVector targets;
      materialRel.GetTargets(&targets);
      if (!targets.empty()) {
        if (auto materialPrim = prim.GetStage()->GetPrimAtPath(targets[0]))
          usdMaterial = pxr::UsdShadeMaterial(materialPrim);
      }
    }

    if (!usdMaterial && binding)
      usdMaterial = binding.ComputeBoundMaterial();

    if (usdMaterial) {
      auto volumeOutput = usdMaterial.GetOutput(pxr::TfToken("nvindex:volume"));
      if (volumeOutput && volumeOutput.HasConnectedSource()) {
        pxr::UsdShadeConnectableAPI source;
        pxr::TfToken sourceName;
        pxr::UsdShadeAttributeType sourceType;
        volumeOutput.GetConnectedSource(&source, &sourceName, &sourceType);
        pxr::UsdShadeShader volumeShader(source.GetPrim());

        if (volumeShader) {
          auto colormapInput = volumeShader.GetInput(pxr::TfToken("colormap"));
          if (colormapInput && colormapInput.HasConnectedSource()) {
            pxr::UsdShadeConnectableAPI colormapSource;
            pxr::TfToken colormapSourceName;
            pxr::UsdShadeAttributeType colormapSourceType;
            if (colormapInput.GetConnectedSource(&colormapSource,
                    &colormapSourceName,
                    &colormapSourceType)) {
              if (auto colormapPrim = colormapSource.GetPrim();
                  colormapPrim && extractColormapFromPrim(colormapPrim, tf))
                return tf;
            }
          }
        }
      }
    }
  }

  // Child Shader prim carrying colormap attributes directly.
  for (const auto &child : prim.GetChildren()) {
    if (!child.IsA<pxr::UsdShadeShader>())
      continue;
    if (extractColormapFromPrim(child, tf))
      return tf;
  }

  return tf;
}

} // namespace

bool isVolumePrimType(const pxr::TfToken &primType)
{
  return primType == pxr::HdPrimTypeTokens->volume;
}

bool convertVolume(ImportContext &ctx,
    const pxr::SdfPath &primPath,
    LayerNodeRef node,
    std::string *skipDetail)
{
  auto prim = ctx.stage->GetPrimAtPath(primPath);
  if (!prim) {
    *skipDetail = "volume has no Stage prim";
    return false;
  }

  const auto primName = primPath.GetString();

  std::vector<std::string> filePaths;
  std::optional<std::string> propertyName;

  auto fieldRel = prim.GetRelationship(pxr::TfToken("field:volume"));
  if (!fieldRel)
    fieldRel = prim.GetRelationship(pxr::TfToken("field:density"));

  if (fieldRel) {
    pxr::SdfPathVector targets;
    fieldRel.GetTargets(&targets);
    if (!targets.empty()) {
      if (auto fieldPrim = ctx.stage->GetPrimAtPath(targets[0])) {
        // Only an unstructured field names the property to read out of the
        // file; every other spatial field format has a single field per file.
        if (fieldPrim.GetTypeName() == "VTUAsset") {
          if (auto attr = fieldPrim.GetAttribute(pxr::TfToken("property"))) {
            std::string value;
            if (attr.Get(&value))
              propertyName = std::move(value);
          }
        }

        if (auto filePathAttr =
                fieldPrim.GetAttribute(pxr::TfToken("filePath"))) {
          std::vector<double> sampleTimes;
          filePathAttr.GetTimeSamples(&sampleTimes);

          auto appendPath = [&](const pxr::SdfAssetPath &assetPath) {
            auto path = assetPath.GetResolvedPath();
            if (path.empty())
              path = assetPath.GetAssetPath();
            if (!path.empty())
              filePaths.push_back(std::move(path));
          };

          if (!sampleTimes.empty()) {
            for (double t : sampleTimes) {
              pxr::SdfAssetPath assetPath;
              if (filePathAttr.Get(&assetPath, t))
                appendPath(assetPath);
            }
          } else {
            pxr::SdfAssetPath assetPath;
            if (filePathAttr.Get(&assetPath))
              appendPath(assetPath);
          }
        }
      }
    }
  }

  if (filePaths.empty()) {
    *skipDetail = "volume names no field file to read";
    return false;
  }

  const auto &filePath = filePaths.front();

  auto field = import_spatial_field(
      *ctx.scene, filePath.c_str(), std::move(propertyName));
  if (!field) {
    *skipDetail = "field file '" + filePath + "' could not be loaded";
    return false;
  }

  const auto tf = getVolumeTransferFunction(prim);
  auto valueRange = field->computeValueRange();

  auto [volumeNode, volume] = ctx.scene->insertNewChildObjectNode<Volume>(
      node, tokens::volume::transferFunction1D);
  volume->setName(primName.c_str());
  volume->setParameterObject("value", *field);

  bool appliedTransferFunction = false;
  if (tf.hasTransferFunction && !tf.colors.empty()) {
    auto coreTF = toTransferFunction(tf);
    if (!coreTF.colorPoints.empty() && !coreTF.opacityPoints.empty()) {
      applyTransferFunction(*ctx.scene, volume, coreTF);
      if (coreTF.range.lower < coreTF.range.upper)
        valueRange = math::float2(coreTF.range.lower, coreTF.range.upper);
      appliedTransferFunction = true;
    }
  }

  if (!appliedTransferFunction) {
    auto colors = makeDefaultColorMap(256);
    auto colorArray = ctx.scene->createArray(ANARI_FLOAT32_VEC4, colors.size());
    colorArray->setData(colors);
    volume->setParameterObject("color", *colorArray);
    volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
  }

  if (auto attr = prim.GetAttribute(pxr::TfToken("anari:valueRange"))) {
    pxr::GfVec2f customRange;
    if (attr.Get(&customRange)) {
      valueRange = math::float2(customRange[0], customRange[1]);
      volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
    }
  }

  float unitDistance = tf.unitDistance;
  if (unitDistance <= 0.0f) {
    if (auto attr = prim.GetAttribute(pxr::TfToken("anari:unitDistance")))
      attr.Get(&unitDistance);
  }
  if (unitDistance > 0.0f)
    volume->setParameter("unitDistance", unitDistance);

  if (filePaths.size() > 1) {
    auto &animation = ctx.animMgr->addAnimation(primName);
    animation.emplaceFileBinding<SpatialFieldFileBinding>(
        ctx.scene, volume.data(), field, std::move(filePaths));
  }

  return true;
}

} // namespace tsd::io::usd
