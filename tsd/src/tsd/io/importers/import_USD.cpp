// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#if TSD_USE_USD
#include "tsd/io/importers/detail/usd/UsdAnimation.h"
#include "tsd/io/importers/detail/usd/UsdDialect.h"
#include "tsd/io/importers/detail/usd/UsdGeometry.h"
#include "tsd/io/importers/detail/usd/UsdImportContext.h"
#include "tsd/io/importers/detail/usd/UsdInstancing.h"
#include "tsd/io/importers/detail/usd/UsdLights.h"
#include "tsd/io/importers/detail/usd/UsdVolume.h"
#include "tsd/io/usd/UsdStageSession.h"
// usd
#include <pxr/imaging/hd/instancedBySchema.h>
#include <pxr/imaging/hd/purposeSchema.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/visibilitySchema.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/xformable.h>
#endif
// std
#include <string>
#include <vector>

namespace tsd::io {

using namespace tsd::core;

#if TSD_USE_USD

namespace {

using namespace tsd::io::usd;

///////////////////////////////////////////////////////////////////////////////
// Traversal //////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

bool purposeIsIncluded(
    const pxr::TfToken &purpose, const UsdPurposeSelection &selection)
{
  if (purpose == pxr::HdRenderTagTokens->guide)
    return selection.guide;
  if (purpose == pxr::HdRenderTagTokens->proxy)
    return selection.proxy;
  if (purpose == pxr::HdRenderTagTokens->render)
    return selection.render;
  return selection.defaultPurpose;
}

// Undo the node a prim created, if any, and leave a named, empty, disabled
// Placeholder Node in its place, so that a gap is visible in the hierarchy
// rather than only in a log. Every prim whose content is a loss goes through
// here, which is what keeps one report entry and one Placeholder Node
// together: converters signal failure, this reports it.
void skipPrim(ImportContext &ctx,
    LayerNodeRef node, // null when the prim never got one
    LayerNodeRef parent,
    const pxr::SdfPath &primPath,
    const std::string &primType,
    UsdSkipReason reason,
    const std::string &detail = "")
{
  if (node)
    ctx.scene->removeNode(node);

  ctx.reportSkip(primPath, primType, reason, detail);

  // insertChildNode() already leaves the node empty; setEmpty() would clear
  // the name along with the value.
  auto placeholder =
      ctx.scene->insertChildNode(parent, primPath.GetName().c_str());
  (*placeholder)->setEnabled(false);
  (*placeholder)->setInstanceParameter("usd:skipReason", Any(toString(reason)));
  (*placeholder)->setInstanceParameter("usd:primPath", Any(primPath.GetText()));
}

struct Traversal
{
  ImportContext *ctx{nullptr};
  pxr::HdSceneIndexBaseRefPtr sceneIndex;
  InstancerRegistry *instancers{nullptr};

  void visit(const pxr::SdfPath &primPath,
      LayerNodeRef parent,
      bool hidden,
      const tsd::math::mat4 &parentXform);

  tsd::math::mat4 localTransformOf(
      const pxr::SdfPath &primPath, bool *resetsXformStack) const;
  bool isHierarchyPrim(const pxr::SdfPath &primPath) const;
};

// Transforms are deliberately not taken from the resolved scene, which
// flattens them: the Stage is retained so the prim hierarchy can be mirrored
// as nested transform nodes with TSD doing the composition.
tsd::math::mat4 Traversal::localTransformOf(
    const pxr::SdfPath &primPath, bool *resetsXformStack) const
{
  *resetsXformStack = false;
  auto prim = ctx->stage->GetPrimAtPath(primPath);
  if (!prim)
    return tsd::math::IDENTITY_MAT4;
  pxr::UsdGeomXformable xformable(prim);
  if (!xformable)
    return tsd::math::IDENTITY_MAT4;
  pxr::GfMatrix4d local(1.0);
  xformable.GetLocalTransformation(&local, resetsXformStack, ctx->importTime);
  return toTsdMat4(local);
}

// A prim with no resolved type is either scene hierarchy (Xform, Scope) or
// something USD cannot image at all; only the latter is a loss worth naming.
bool Traversal::isHierarchyPrim(const pxr::SdfPath &primPath) const
{
  auto prim = ctx->stage->GetPrimAtPath(primPath);
  return !prim || bool(pxr::UsdGeomImageable(prim));
}

void Traversal::visit(const pxr::SdfPath &primPath,
    LayerNodeRef parent,
    bool hidden,
    const tsd::math::mat4 &parentXform)
{
  // Claimed Prims reach the Scene through the dialect's own importers; the
  // generic path must not also convert a carrier prim into geometry.
  if (ctx->isClaimed(primPath))
    return;

  auto prim = sceneIndex->GetPrim(primPath);

  // Prototype content reaches the Scene through its instancer, not here.
  auto instancedBy = pxr::HdInstancedBySchema::GetFromParent(prim.dataSource);
  if (auto paths = instancedBy.GetPaths()) {
    if (!paths->GetTypedValue(0).empty())
      return;
  }

  // Purpose //

  pxr::TfToken purpose = pxr::HdRenderTagTokens->geometry;
  if (auto purposeSchema =
          pxr::HdPurposeSchema::GetFromParent(prim.dataSource)) {
    if (auto value = purposeSchema.GetPurpose())
      purpose = value->GetTypedValue(0);
  }
  if (!purposeIsIncluded(purpose, ctx->options->purposes)) {
    skipPrim(*ctx,
        {},
        parent,
        primPath,
        prim.primType.GetString(),
        UsdSkipReason::PURPOSE_EXCLUDED,
        purpose.GetString());
    return;
  }

  // Visibility //

  bool visible = true;
  if (auto visibilitySchema =
          pxr::HdVisibilitySchema::GetFromParent(prim.dataSource)) {
    if (auto value = visibilitySchema.GetVisibility())
      visible = value->GetTypedValue(0);
  }
  if (!visible && !hidden) {
    ctx->reportSkip(
        primPath, prim.primType.GetString(), UsdSkipReason::RESOLVED_INVISIBLE);
  }

  // Visibility is imported as one static enabled/disabled state, so say when
  // the Stage animates it rather than leaving the difference to be noticed.
  // Exporters routinely re-author every attribute at every frame, so the
  // samples are compared: a value that never changes is not a loss.
  if (auto imageable =
          pxr::UsdGeomImageable(ctx->stage->GetPrimAtPath(primPath))) {
    if (attributeValueVaries(imageable.GetVisibilityAttr())) {
      ctx->reportSkip(primPath,
          prim.primType.GetString(),
          UsdSkipReason::TIME_VARYING_VALUE_DROPPED,
          "visibility is time-sampled; imported at the Stage's start of time");
    }
  }
  const bool subtreeHidden = hidden || !visible;

  // Node for this prim //

  bool resetsXformStack = false;
  const auto localXform = localTransformOf(primPath, &resetsXformStack);

  // A prim that resets the transform stack ignores its ancestors in USD. The
  // node stays where its name belongs in the hierarchy and cancels the
  // accumulated ancestor transform instead, so TSD's own composition lands on
  // the same place USD does.
  const auto nodeXform = resetsXformStack
      ? tsd::math::mul(tsd::math::inverse(parentXform), localXform)
      : localXform;
  const auto accumulatedXform = tsd::math::mul(parentXform, nodeXform);

  auto node = ctx->scene->insertChildTransformNode(
      parent, nodeXform, primPath.GetName().c_str());
  if (resetsXformStack)
    (*node)->setInstanceParameter("usd:resetXformStack", Any(true));
  if (subtreeHidden)
    (*node)->setEnabled(false);

  // Content //

  bool convertedAnything = false;
  if (prim.primType.IsEmpty()) {
    if (!isHierarchyPrim(primPath)) {
      skipPrim(*ctx,
          node,
          parent,
          primPath,
          ctx->stage->GetPrimAtPath(primPath).GetTypeName().GetString(),
          UsdSkipReason::UNSUPPORTED_PRIM_TYPE);
      return;
    }
  } else if (isGeometryPrimType(prim.primType)) {
    auto converted = convertGeometry(
        *ctx, sceneIndex, primPath, prim, tsd::math::IDENTITY_MAT4);
    for (auto &surface : converted.surfaces)
      ctx->scene->insertChildObjectNode(node, surface, surface->name().c_str());
    addDeformingGeometryAnimation(*ctx, primPath, converted);
    convertedAnything = true;
  } else if (prim.primType == pxr::HdPrimTypeTokens->instancer) {
    convertInstancer(*ctx, sceneIndex, primPath, prim, node, *instancers);
    convertedAnything = true;
  } else if (isLightPrimType(prim.primType)) {
    std::string skipDetail;
    if (auto light = convertLight(*ctx, primPath, prim, &skipDetail)) {
      ctx->scene->insertChildObjectNode(
          node, light, primPath.GetName().c_str());
      convertedAnything = true;
    } else {
      skipPrim(*ctx,
          node,
          parent,
          primPath,
          prim.primType.GetString(),
          UsdSkipReason::UNSUPPORTED_LIGHT_TYPE,
          skipDetail);
      return;
    }
  } else if (prim.primType == pxr::HdPrimTypeTokens->camera) {
    convertCamera(*ctx, primPath);
    convertedAnything = true;
  } else if (prim.primType == pxr::HdPrimTypeTokens->material
      || prim.primType == pxr::HdPrimTypeTokens->geomSubset) {
    // Materials convert on demand from the prims that bind them; geom subsets
    // are consumed by their parent mesh. Neither is a loss.
    ctx->scene->removeNode(node);
    return;
  } else if (isVolumePrimType(prim.primType)) {
    std::string skipDetail;
    convertedAnything = convertVolume(*ctx, primPath, node, &skipDetail);
    if (!convertedAnything) {
      skipPrim(*ctx,
          node,
          parent,
          primPath,
          prim.primType.GetString(),
          UsdSkipReason::FIELD_LOAD_FAILED,
          skipDetail);
      return;
    }
  } else {
    skipPrim(*ctx,
        node,
        parent,
        primPath,
        prim.primType.GetString(),
        UsdSkipReason::UNSUPPORTED_PRIM_TYPE);
    return;
  }

  if (convertedAnything)
    ctx->report->convertedPrims++;

  instancers->recordNode(primPath, node);
  addTransformAnimation(*ctx, primPath, node);

  for (const auto &childPath : sceneIndex->GetChildPrimPaths(primPath))
    visit(childPath, node, subtreeHidden, accumulatedXform);
}

} // namespace

UsdImportReport import_USD(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filepath,
    LayerNodeRef location,
    const UsdImportOptions &options)
{
  UsdImportReport report;

  // The Session owns the Stage and the chain that resolves it; every animation
  // binding this import creates joins the same one, so a scrub resolves
  // through exactly what was converted here. A fully static import lets go of
  // it on return.
  auto session = usd::acquireUsdSession(filepath);
  if (!session) {
    logError("[import_USD] failed to open stage '%s'", filepath);
    return report;
  }
  report.stageOpened = true;

  auto stage = session->stage();
  ImportContext ctx{&scene,
      &animMgr,
      &options,
      &report,
      session,
      stage,
      filepath,
      pathOf(filepath)};

  // Values authored only as time samples do not resolve at UsdTimeCode's
  // default, so the import reads at the Stage's own start of time instead.
  ctx.importTime = pxr::UsdTimeCode(session->startTimeCode());
  session->setTime(ctx.importTime);

  // Dialect pre-pass: markers on the raw Stage claim whole subtrees, which the
  // traversal skips so the generic path never converts a carrier prim into
  // meaningless geometry.
  auto claimed = claimDialectPrims(ctx);
  ctx.claimedPrims = claimed.get();

  auto sceneIndex = session->sceneIndex();

  auto root = scene.insertChildNode(
      location ? location : scene.defaultLayer()->root(), filepath);

  // Record how the Stage is framed for the application to consume. No
  // corrective root transform is inserted: coordinates stay comparable to the
  // Stage and dome lights keep their own orientation.
  (*root)->setInstanceParameter(
      "usd:upAxis", Any(pxr::UsdGeomGetStageUpAxis(stage).GetText()));
  (*root)->setInstanceParameter("usd:metersPerUnit",
      Any(float(pxr::UsdGeomGetStageMetersPerUnit(stage))));

  const pxr::SdfPath scopeRoot = options.primPath.empty()
      ? pxr::SdfPath::AbsoluteRootPath()
      : pxr::SdfPath(options.primPath);

  InstancerRegistry instancers(sceneIndex);
  Traversal traversal{&ctx, sceneIndex, &instancers};

  scene.beginLayerEditBatch();
  if (scopeRoot == pxr::SdfPath::AbsoluteRootPath()) {
    for (const auto &childPath : sceneIndex->GetChildPrimPaths(scopeRoot)) {
      if (childPath.GetString() == NATIVE_INSTANCING_ROOT)
        continue;
      traversal.visit(childPath, root, false, tsd::math::IDENTITY_MAT4);
    }
  } else {
    traversal.visit(scopeRoot, root, false, tsd::math::IDENTITY_MAT4);
  }

  // Native-instance placements live outside the mirrored hierarchy; attach
  // their shared Prototype objects at each placement's own node.
  attachNativeInstances(ctx, sceneIndex, instancers, root);

  // Dialect content is routed to the handlers that already know these formats.
  importDialectPrims(ctx, sceneIndex, claimed, root);
  scene.endLayerEditBatch();

  logStatus("[import_USD] %s: %s", filepath, report.summary().c_str());

  return report;
}

#else

UsdImportReport import_USD(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filepath,
    LayerNodeRef location,
    const UsdImportOptions &options)
{
  logError("[import_USD] USD not enabled in TSD build.");
  return {};
}

#endif

} // namespace tsd::io
