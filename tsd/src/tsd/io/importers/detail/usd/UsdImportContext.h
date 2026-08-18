// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Logging.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/io/UsdImport.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/io/usd/UsdStageSession.h"
#include "tsd/scene/Scene.hpp"
// usd
#include <pxr/base/gf/matrix4d.h>
#include <pxr/imaging/hd/dataSourceTypeDefs.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
// std
#include <memory>
#include <string>
#include <unordered_map>

namespace tsd::animation {
struct Animation;
struct AnimationManager;
} // namespace tsd::animation

namespace tsd::io::usd {

using namespace tsd::scene;

/*
 * Everything one USD Stage import needs to carry between converters: the
 * target Scene, the settings driving the import, the report being accumulated,
 * and the Stage itself, which is retained so that data OpenUSD does not model
 * -- the `anari:` and `tsd:io:` attribute vocabularies, carrier metadata --
 * can be read directly from prims by path.
 *
 * Example:
 *   ImportContext ctx{scene, animMgr, options, report, stage, filename};
 *   ctx.reportSkip(primPath, "cylinderLight",
 *       UsdSkipReason::UNSUPPORTED_LIGHT_TYPE);
 */
struct ImportContext
{
  Scene &scene;
  tsd::animation::AnimationManager &animMgr;
  const UsdImportOptions &options;
  UsdImportReport &report;
  std::shared_ptr<UsdStageSession> session;
  pxr::UsdStageRefPtr stage;
  std::string filePath;
  std::string basePath;

  // The time everything static is read at. Deliberately not
  // UsdTimeCode::Default(), at which values authored only as time samples do
  // not resolve at all.
  pxr::UsdTimeCode importTime{pxr::UsdTimeCode::EarliestTime()};

  // One Import is one Animation (ADR 0009), created on the first binding that
  // needs it and named for the Stage's file. Per-prim Animations collided on
  // leaf names and implied independent control that does not exist: every
  // Animation is driven by the same AnimationManager clock.
  tsd::animation::Animation &animation();

  // Fold a bound attribute's authored sample count into the Import Report.
  void reportSampleCount(size_t count);

  // Set by animation(); this stays an aggregate, so it cannot be private.
  tsd::animation::Animation *importAnimation{nullptr};

  // Caches keyed by resolved prim path, so shared content converts once.
  ImageCache textureCache{&scene};
  std::unordered_map<std::string, MaterialRef> materialCache;
  std::unordered_map<std::string, std::string> uvPrimvarCache;

  void reportSkip(const pxr::SdfPath &primPath,
      const std::string &primType,
      UsdSkipReason reason,
      const std::string &detail = "");
};

// Small conversions shared by every converter /////////////////////////////////

tsd::math::mat4 toTsdMat4(const pxr::GfMatrix4d &m);
pxr::VtIntArray intArrayOf(const pxr::HdIntArrayDataSourceHandle &source);

// Whether an attribute's time samples actually differ from one another. USD
// exporters routinely re-author every attribute at every frame regardless of
// change, so "is time-sampled" is not the same question as "is animated".
//
// The comparison is deliberately asymmetric: a time-sampled *array* attribute
// is assumed to vary without reading it, because proving otherwise means
// reading every sample -- gigabytes for a particle simulation. So a large array
// authored identically at every frame is still treated as animated, still gets
// a binding, and is still re-pulled per frame.
bool attributeValueVaries(const pxr::UsdAttribute &attribute);

// Inlined definitions ////////////////////////////////////////////////////////

inline void ImportContext::reportSkip(const pxr::SdfPath &primPath,
    const std::string &primType,
    UsdSkipReason reason,
    const std::string &detail)
{
  report.skipped.push_back({primPath.GetString(), primType, reason, detail});
  core::logStatus("[import_USD] %s: %s%s%s",
      primPath.GetText(),
      toString(reason),
      detail.empty() ? "" : " -- ",
      detail.c_str());
}

inline pxr::VtIntArray intArrayOf(const pxr::HdIntArrayDataSourceHandle &source)
{
  return source ? source->GetTypedValue(0) : pxr::VtIntArray();
}

inline tsd::math::mat4 toTsdMat4(const pxr::GfMatrix4d &m)
{
  tsd::math::mat4 retval;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      retval[i][j] = static_cast<float>(m[i][j]);
  return retval;
}

} // namespace tsd::io::usd
