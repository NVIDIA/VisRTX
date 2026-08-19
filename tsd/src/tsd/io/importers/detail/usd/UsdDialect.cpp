// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdDialect.h"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/usd/UsdMaterials.h"
// usd
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/usd/usd/collectionAPI.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
// std
#include <string>
#include <vector>

namespace tsd::io::usd {

using namespace tsd::core;

namespace {

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

// The binding is computed on the raw Stage rather than read from the resolved
// scene because a carrier prim is a Claimed Prim: the traversal never visits
// it, so nothing has asked for its material. Being claimed does not put the
// Material prim out of reach, though -- a claim only suppresses traversal,
// while resolveMaterial() reads the resolved scene by path -- so the material
// converts through the same converter every other binding goes through, which
// is what lets EnSight parts share materials with the rest of the import.
MaterialRef boundMaterialOf(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::UsdPrim &prim)
{
  pxr::UsdShadeMaterialBindingAPI binding(prim);
  if (!binding)
    return {};
  auto usdMaterial = binding.ComputeBoundMaterial();
  if (!usdMaterial)
    return {};
  return resolveMaterial(ctx, sceneIndex, usdMaterial.GetPath()).material;
}

void importEnsightDataset(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
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

  auto fallbackMaterial = boundMaterialOf(ctx, sceneIndex, scopePrim);
  core::FlatMap<std::string, MaterialRef> perPartMaterials;
  for (const auto &child : scopePrim.GetChildren()) {
    auto childMaterial = boundMaterialOf(ctx, sceneIndex, child);
    if (childMaterial && childMaterial != fallbackMaterial)
      perPartMaterials[child.GetName().GetString()] = childMaterial;
  }

  auto scopeNode = ctx.scene->insertChildNode(parent, primName.c_str());
  import_ENSIGHT(*ctx.scene,
      *ctx.animMgr,
      caseFile.c_str(),
      scopeNode,
      fields,
      datasetSettings.root(),
      fallbackMaterial,
      perPartMaterials,
      0);
  ctx.report->convertedPrims++;
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
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
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
          ctx, sceneIndex, prim, importRoot, claimed->renderSettings.root());
      break;
    }
  }
}

} // namespace tsd::io::usd
