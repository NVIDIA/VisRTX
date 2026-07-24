// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Logging.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/io/UsdImport.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/scene/Scene.hpp"
// usd
#include <pxr/base/gf/matrix4d.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
// std
#include <string>
#include <unordered_map>

namespace tsd::animation {
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
  pxr::UsdStageRefPtr stage;
  std::string filePath;
  std::string basePath;

  // The time everything static is read at. Deliberately not
  // UsdTimeCode::Default(), at which values authored only as time samples do
  // not resolve at all.
  pxr::UsdTimeCode importTime{pxr::UsdTimeCode::EarliestTime()};

  // Caches keyed by resolved prim path, so shared content converts once.
  TextureCache textureCache;
  std::unordered_map<std::string, MaterialRef> materialCache;
  std::unordered_map<std::string, std::string> uvPrimvarCache;

  void reportSkip(const pxr::SdfPath &primPath,
      const std::string &primType,
      UsdSkipReason reason,
      const std::string &detail = "");
};

// Small conversions shared by every converter /////////////////////////////////

tsd::math::mat4 toTsdMat4(const pxr::GfMatrix4d &m);

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

inline tsd::math::mat4 toTsdMat4(const pxr::GfMatrix4d &m)
{
  tsd::math::mat4 retval;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      retval[i][j] = static_cast<float>(m[i][j]);
  return retval;
}

} // namespace tsd::io::usd
