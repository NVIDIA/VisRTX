// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdDialect.h"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/io/animation/SpatialFieldFileBinding.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/usd/UsdMaterials.h"
#include "tsd/scene/objects/Array.hpp"
// usd
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usd/collectionAPI.h>
#include <pxr/usd/usd/primRange.h>
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

///////////////////////////////////////////////////////////////////////////////
// Transfer functions authored on a Stage ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// Render settings and EnSight carriers ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void readRenderSettings(
    const pxr::UsdStageRefPtr &stage, core::DataNode &settings)
{
  for (const auto &prim : stage->Traverse()) {
    if (prim.GetTypeName() != "RenderSettings")
      continue;

    if (auto attr = prim.GetAttribute(pxr::TfToken("tsd:io:cutPlane"))) {
      pxr::GfVec4f value;
      if (attr.Get(&value))
        settings["cutPlane"] =
            math::float4(value[0], value[1], value[2], value[3]);
    }

    if (auto collection = pxr::UsdCollectionAPI::Get(
            prim, pxr::TfToken("tsd:io:cutPlaneTarget"))) {
      pxr::SdfPathVector includes;
      collection.GetIncludesRel().GetTargets(&includes);
      auto &targets = settings["cutPlaneTargets"];
      for (const auto &path : includes)
        targets.append() = std::string(path.GetString());
    }

    break; // only the first RenderSettings prim
  }
}

bool primIsEnsightCarrier(const pxr::UsdPrim &prim)
{
  if (!prim.GetChildren())
    return false;
  auto firstChild = *prim.GetChildren().begin();
  return firstChild && firstChild.GetCustomData().count("ensight") > 0;
}

std::string ensightCaseFileOf(const pxr::UsdPrim &scopePrim)
{
  for (const auto &child : scopePrim.GetChildren()) {
    for (const auto &spec : child.GetPrimStack()) {
      auto customLayerData = spec->GetLayer()->GetCustomLayerData();
      auto found = customLayerData.find("ensight");
      if (found == customLayerData.end())
        continue;
      const auto &dictionary = found->second.Get<pxr::VtDictionary>();
      auto caseFile = dictionary.find("caseFile");
      if (caseFile != dictionary.end())
        return caseFile->second.Get<std::string>();
    }
  }
  return {};
}

// Resolve a material bound on a raw Stage prim, going through the same
// converter the resolved path uses so EnSight parts share materials with the
// rest of the import.
MaterialRef boundMaterialOf(ImportContext &ctx, const pxr::UsdPrim &prim)
{
  pxr::UsdShadeMaterialBindingAPI binding(prim);
  if (!binding)
    return {};
  auto usdMaterial = binding.ComputeBoundMaterial();
  if (!usdMaterial)
    return {};

  const auto key = usdMaterial.GetPath().GetString();
  if (auto found = ctx.materialCache.find(key);
      found != ctx.materialCache.end())
    return found->second.material;
  return {};
}

void importEnsightDataset(ImportContext &ctx,
    const pxr::UsdPrim &scopePrim,
    LayerNodeRef parent,
    const core::DataNode &settings)
{
  const auto primName = scopePrim.GetName().GetString();
  const auto caseFile = ensightCaseFileOf(scopePrim);
  if (caseFile.empty()) {
    logWarning("[import_USD] EnSight scope '%s': no case file found",
        primName.c_str());
    return;
  }

  std::vector<std::string> fields;
  for (int i = 0; i < 4; ++i) {
    const auto attrName = "ensight:fieldMapping:attribute" + std::to_string(i);
    auto attr = scopePrim.GetAttribute(pxr::TfToken(attrName));
    if (!attr)
      continue;
    std::string varName;
    if (attr.Get(&varName) && !varName.empty())
      fields.push_back(varName);
  }

  const auto primPath = scopePrim.GetPath().GetString();
  core::DataTree datasetSettings;
  const auto *targets = settings.child("cutPlaneTargets");
  const auto *cutPlane = settings.child("cutPlane");
  if (cutPlane && targets) {
    for (size_t i = 0; i < targets->numChildren(); ++i) {
      const auto target = targets->child(i)->getValueAs<std::string>();
      if (target == primPath) {
        datasetSettings.root()["cutPlane"] = cutPlane->getValue();
        datasetSettings.root().remove("cutPlaneTargets");
        break;
      } else if (target.substr(0, primPath.size() + 1) == primPath + "/") {
        datasetSettings.root()["cutPlane"] = cutPlane->getValue();
        datasetSettings.root()["cutPlaneTarget"].append(
            target.substr(primPath.size() + 1));
      }
    }
  }

  auto fallbackMaterial = boundMaterialOf(ctx, scopePrim);
  core::FlatMap<std::string, MaterialRef> perPartMaterials;
  for (const auto &child : scopePrim.GetChildren()) {
    auto childMaterial = boundMaterialOf(ctx, child);
    if (childMaterial && childMaterial != fallbackMaterial)
      perPartMaterials[child.GetName().GetString()] = childMaterial;
  }

  auto scopeNode = ctx.scene.insertChildNode(parent, primName.c_str());
  import_ENSIGHT(ctx.scene,
      ctx.animMgr,
      caseFile.c_str(),
      scopeNode,
      fields,
      datasetSettings.root(),
      fallbackMaterial,
      perPartMaterials,
      0);
  ctx.report.convertedPrims++;
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
// Claim and prune ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::shared_ptr<ClaimedPrims> claimDialectPrims(ImportContext &ctx)
{
  auto retval = std::make_shared<ClaimedPrims>();
  readRenderSettings(ctx.stage, retval->renderSettings.root());

  for (const auto &prim : ctx.stage->Traverse()) {
    if (primIsEnsightCarrier(prim)) {
      retval->entries.push_back(
          {prim.GetPath(), ClaimedPrims::Kind::ENSIGHT_DATASET});
    }
  }

  return retval;
}

bool ClaimedPrims::claims(const pxr::SdfPath &path) const
{
  for (const auto &entry : entries) {
    if (path == entry.path || path.HasPrefix(entry.path))
      return true;
  }
  return false;
}

void importDialectPrims(ImportContext &ctx,
    const std::shared_ptr<ClaimedPrims> &claimed,
    LayerNodeRef importRoot)
{
  if (!claimed)
    return;

  for (const auto &entry : claimed->entries) {
    auto prim = ctx.stage->GetPrimAtPath(entry.path);
    if (!prim)
      continue;
    switch (entry.kind) {
    case ClaimedPrims::Kind::ENSIGHT_DATASET:
      importEnsightDataset(
          ctx, prim, importRoot, claimed->renderSettings.root());
      break;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Volumes ////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

bool isVolumePrimType(const pxr::TfToken &primType)
{
  return primType == pxr::HdPrimTypeTokens->volume;
}

bool convertVolume(
    ImportContext &ctx, const pxr::SdfPath &primPath, LayerNodeRef node)
{
  auto prim = ctx.stage->GetPrimAtPath(primPath);
  if (!prim)
    return false;

  const auto primName = primPath.GetString();

  std::vector<std::string> filePaths;
  bool isVtuAsset = false;
  std::optional<std::string> propertyName;

  auto fieldRel = prim.GetRelationship(pxr::TfToken("field:volume"));
  if (!fieldRel)
    fieldRel = prim.GetRelationship(pxr::TfToken("field:density"));

  if (fieldRel) {
    pxr::SdfPathVector targets;
    fieldRel.GetTargets(&targets);
    if (!targets.empty()) {
      if (auto fieldPrim = ctx.stage->GetPrimAtPath(targets[0])) {
        isVtuAsset = fieldPrim.GetTypeName() == "VTUAsset";
        if (isVtuAsset) {
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

  if (filePaths.empty())
    return false;

  const auto &filePath = filePaths.front();

  SpatialFieldRef field;
  if (isVtuAsset) {
    field = import_spatial_field(
        ctx.scene, filePath.c_str(), std::move(propertyName));
  } else {
    const auto extension = extensionOf(filePath);
    if (extension == ".raw")
      field = import_RAW(ctx.scene, filePath.c_str());
    else if (extension == ".flash")
      field = import_FLASH(ctx.scene, filePath.c_str());
    else if (extension == ".nvdb" || extension == ".vdb")
      field = import_NVDB(ctx.scene, filePath.c_str());
    else if (extension == ".mhd")
      field = import_MHD(ctx.scene, filePath.c_str());
    else if (extension == ".vtu")
      field = import_VTU(ctx.scene, filePath.c_str(), propertyName);
  }

  if (!field)
    return false;

  const auto tf = getVolumeTransferFunction(prim);
  auto valueRange = field->computeValueRange();

  auto [volumeNode, volume] = ctx.scene.insertNewChildObjectNode<Volume>(
      node, tokens::volume::transferFunction1D);
  volume->setName(primName.c_str());
  volume->setParameterObject("value", *field);

  bool appliedTransferFunction = false;
  if (tf.hasTransferFunction && !tf.colors.empty()) {
    auto coreTF = toTransferFunction(tf);
    if (!coreTF.colorPoints.empty() && !coreTF.opacityPoints.empty()) {
      applyTransferFunction(ctx.scene, volume, coreTF);
      if (coreTF.range.lower < coreTF.range.upper)
        valueRange = math::float2(coreTF.range.lower, coreTF.range.upper);
      appliedTransferFunction = true;
    }
  }

  if (!appliedTransferFunction) {
    auto colors = makeDefaultColorMap(256);
    auto colorArray = ctx.scene.createArray(ANARI_FLOAT32_VEC4, colors.size());
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
    auto &animation = ctx.animMgr.addAnimation(primName);
    animation.emplaceFileBinding<SpatialFieldFileBinding>(
        &ctx.scene, volume.data(), field, std::move(filePaths));
  }

  return true;
}

} // namespace tsd::io::usd
