// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Logging.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/io/UsdImport.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/io/usd/UsdDataSource.h"
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
#include <utility>

namespace tsd::animation {
struct Animation;
struct AnimationManager;
} // namespace tsd::animation

namespace tsd::io::usd {

using namespace tsd::scene;

struct ClaimedPrims;

/*
 * A material as TSD sees it, together with the primvar its own texture-reader
 * node asked for. The UV name travels with the material because the geometry
 * converter must bind that primvar rather than assume a conventional name.
 *
 * A default-constructed value is a material that did not resolve, which is
 * cached like any other so a Stage binding one broken material a thousand
 * times resolves and reports it once.
 */
struct ResolvedMaterial
{
  MaterialRef material;
  std::string uvPrimvarName;
};

/*
 * Everything one USD Stage import needs to carry between converters: the
 * target Scene, the settings driving the import, the report being accumulated,
 * and the Stage itself, which is retained so that data OpenUSD does not model
 * -- the `anari:` and `tsd:io:` attribute vocabularies, carrier metadata --
 * can be read directly from prims by path.
 *
 * One import makes one of these and passes it by reference from there on: it
 * is not copyable, because `textureCache` holds the same Scene the `scene`
 * member names and a copy is the only way the two could ever come to name
 * different Scenes -- which would put an image's Sampler somewhere the rest
 * of the import never reached.
 *
 * Example:
 *   ImportContext ctx{&scene, &animMgr, &options, &report,
 *       session, stage, filePath, basePath};
 *   ctx.reportSkip(primPath, "cylinderLight",
 *       UsdSkipReason::UNSUPPORTED_LIGHT_TYPE);
 */
struct ImportContext
{
  ImportContext(Scene *scene,
      tsd::animation::AnimationManager *animMgr,
      const UsdImportOptions *options,
      UsdImportReport *report,
      std::shared_ptr<UsdStageSession> session,
      pxr::UsdStageRefPtr stage,
      std::string filePath,
      std::string basePath);

  ImportContext(const ImportContext &) = delete;
  ImportContext &operator=(const ImportContext &) = delete;

  Scene *scene{nullptr};
  tsd::animation::AnimationManager *animMgr{nullptr};
  const UsdImportOptions *options{nullptr};
  UsdImportReport *report{nullptr};
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

  // Record one prim as animated, with the number of time samples the binding
  // was built from, so the Import Report can name the Stage's frame range.
  void reportAnimatedPrim(size_t sampleCount);

  // Prims the TSD dialect claimed, which every path that walks the resolved
  // scene must skip: they reach the Scene through the dialect's own importers.
  // Set by the dialect pre-pass; null until then. Kept here rather than in the
  // resolution chain so the Stage Session stays free of one Import's handling.
  const ClaimedPrims *claimedPrims{nullptr};
  bool isClaimed(const pxr::SdfPath &path) const;

  // Set by animation(). An index rather than a pointer or reference: the
  // AnimationManager holds its Animations by value in a vector, so any other
  // addAnimation() during this import -- a camera's, the dialect's -- moves
  // the one this import made.
  static constexpr size_t NO_ANIMATION = ~size_t(0);
  size_t importAnimationIndex{NO_ANIMATION};

  // Caches keyed by resolved prim path, so shared content converts once.
  ImageCache textureCache{scene};
  std::unordered_map<std::string, ResolvedMaterial> materialCache;

  void reportSkip(const pxr::SdfPath &primPath,
      const std::string &primType,
      UsdSkipReason reason,
      const std::string &detail = "");
};

// Small conversions shared by every converter /////////////////////////////////

tsd::math::mat4 toTsdMat4(const pxr::GfMatrix4d &m);

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

inline ImportContext::ImportContext(Scene *scene,
    tsd::animation::AnimationManager *animMgr,
    const UsdImportOptions *options,
    UsdImportReport *report,
    std::shared_ptr<UsdStageSession> session,
    pxr::UsdStageRefPtr stage,
    std::string filePath,
    std::string basePath)
    : scene(scene),
      animMgr(animMgr),
      options(options),
      report(report),
      session(std::move(session)),
      stage(std::move(stage)),
      filePath(std::move(filePath)),
      basePath(std::move(basePath))
{}

inline void ImportContext::reportSkip(const pxr::SdfPath &primPath,
    const std::string &primType,
    UsdSkipReason reason,
    const std::string &detail)
{
  report->skipped.push_back({primPath.GetString(), primType, reason, detail});
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
