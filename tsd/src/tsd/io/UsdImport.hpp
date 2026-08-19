// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/DataTree.hpp"
// std
#include <string>
#include <vector>

namespace tsd::animation {
struct AnimationManager;
} // namespace tsd::animation

namespace tsd::io {

///////////////////////////////////////////////////////////////////////////////
// Import options /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
 * Which USD Purposes a Stage import includes. TSD deliberately deviates from
 * reference-viewer defaults by including render Purpose (ADR 0017), because
 * assets whose real content sits behind a render Purpose would otherwise
 * import as bounding-box stand-ins.
 *
 * Example:
 *   UsdPurposeSelection p; // default + render
 *   p.proxy = true;        // also inspect proxy stand-ins
 */
struct UsdPurposeSelection
{
  bool defaultPurpose{true};
  bool render{true};
  bool proxy{false};
  bool guide{false};
};

/*
 * How USD materials are emitted into the Scene. Portable physically-based
 * materials are the default so that imported Stages render on every ANARI
 * device; the native passthrough modes trade portability for fidelity.
 */
enum class UsdMaterialMode
{
  PHYSICALLY_BASED,
  MATERIALX,
  MDL
};

const char *toString(UsdMaterialMode mode);
UsdMaterialMode usdMaterialModeFromString(const std::string &name);

/*
 * Typed settings for one USD Stage import. Every field has a default that
 * makes the common case need no configuration, and the whole value converts to
 * and from TSD's data-tree representation so applications can persist it and
 * scripting can drive it.
 *
 * Example:
 *   UsdImportOptions opts;
 *   opts.purposes.proxy = true;
 *   opts.primPath = "/World/Asset";
 *   auto report = import_USD(scene, animMgr, file, {}, opts);
 */
struct UsdImportOptions
{
  UsdPurposeSelection purposes;

  // Ordered Render Context preference with per-material fallback. Entries are
  // USD Render Context names; the empty string is the universal context.
  std::vector<std::string> renderContexts{"", "glslfx"};

  UsdMaterialMode materialMode{UsdMaterialMode::PHYSICALLY_BASED};

  // Subdivision refinement level. TSD refines by default (ADR 0017) rather
  // than matching a reference viewer's unrefined complexity.
  int refinementLevel{2};

  // Import one subtree instead of everything beneath the pseudo-root. Empty
  // imports the whole Stage.
  std::string primPath;

  void toDataNode(core::DataNode &node) const;
  void fromDataNode(const core::DataNode &node);
};

///////////////////////////////////////////////////////////////////////////////
// Import report //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
 * Why a prim did not become renderable TSD content. Recorded per prim in the
 * Import Report and tagged onto the Placeholder Node left in the prim's place.
 */
enum class UsdSkipReason
{
  PURPOSE_EXCLUDED,
  RESOLVED_INVISIBLE,
  UNSUPPORTED_PRIM_TYPE,
  MATERIAL_RESOLUTION_FAILED,
  TEXTURE_LOAD_FAILED,
  FIELD_LOAD_FAILED,
  UNSUPPORTED_LIGHT_TYPE,
  RICHER_MATERIAL_AVAILABLE,
  TIME_VARYING_VALUE_DROPPED,

  // Not a reason: the count of reasons above, so that adding one does not
  // require updating a second list.
  COUNT
};

const char *toString(UsdSkipReason reason);

/*
 * One prim that did not become renderable TSD content, together with why.
 */
struct UsdSkippedPrim
{
  std::string primPath;
  std::string primType;
  UsdSkipReason reason{UsdSkipReason::UNSUPPORTED_PRIM_TYPE};
  std::string detail;
};

/*
 * What an import did. Returned from the import entry point rather than only
 * logged, which is what turns "content was silently dropped" into an
 * assertable condition.
 *
 * Note that a prim can appear in `skipped` and still have produced a Layer
 * node: a prim resolving to invisible imports its real content as a disabled
 * node so it can be toggled on, and is reported because it does not render.
 * Prims skipped for any other reason leave an empty, disabled Placeholder Node
 * at their position in the hierarchy.
 *
 * Example:
 *   auto report = import_USD(scene, animMgr, file);
 *   REQUIRE(report.skipped.empty());
 */
struct UsdImportReport
{
  bool stageOpened{false};
  size_t convertedPrims{0};
  std::vector<UsdSkippedPrim> skipped;

  // How many prims got an animation binding, in place of the per-prim list an
  // Import used to leave behind when every animated prim got its own
  // Animation.
  size_t animatedPrims{0};

  // What the Stage's own clock says, for the application to fold into the one
  // playback clock every animation shares. Reported, never applied: an Import
  // does not reach into global playback state. `sampleCount` is the largest
  // number of authored time samples any bound attribute carries -- zero when
  // the Import found no animation.
  size_t sampleCount{0};
  float timeCodesPerSecond{0.f};

  size_t countOf(UsdSkipReason reason) const;
  bool contains(UsdSkipReason reason) const;
  std::string summary() const;
};

// Fold what a USD Import reported about its Stage's clock into the one
// playback clock every Animation shares, widening it and never shrinking it.
// This is deliberately not done by the Import itself: an Import reports the
// Stage's frame range and rate, and the application decides what to do with
// them -- SciVis Studio, for one, keeps its shot authoritative. A conflict
// between two Stages is logged and the larger value wins.
void widenAnimationClock(
    tsd::animation::AnimationManager &animMgr, const UsdImportReport &report);

} // namespace tsd::io
