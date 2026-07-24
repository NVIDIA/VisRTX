// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/DataTree.hpp"
#include "tsd/io/importers/detail/usd/UsdImportContext.h"
// usd
#include <pxr/imaging/hd/sceneIndex.h>
// std
#include <memory>
#include <vector>

namespace tsd::io::usd {

/*
 * Prims the TSD dialect owns. A pre-pass scans the raw Stage for dialect
 * markers and collects the prim paths they claim; those subtrees are pruned
 * from the resolved scene so the generic path never sees them, and are routed
 * to the handlers that already know these formats. This is what stops an
 * EnSight carrier prim -- which is a mesh prim -- from also being converted
 * into meaningless geometry.
 */
struct ClaimedPrims
{
  enum class Kind
  {
    ENSIGHT_DATASET
  };

  struct Entry
  {
    pxr::SdfPath path;
    Kind kind;
  };

  std::vector<Entry> entries;
  core::DataTree renderSettings;
};

// Scan the raw Stage for dialect markers.
std::shared_ptr<ClaimedPrims> claimDialectPrims(ImportContext &ctx);

// Remove every Claimed Prim subtree from the resolved scene.
pxr::HdSceneIndexBaseRefPtr pruneClaimedPrims(
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const std::shared_ptr<ClaimedPrims> &claimed);

// Route Claimed Prims to the dialect's own importers.
void importDialectPrims(ImportContext &ctx,
    const std::shared_ptr<ClaimedPrims> &claimed,
    LayerNodeRef importRoot);

bool isVolumePrimType(const pxr::TfToken &primType);

// Import a UsdVol Volume prim, honouring the `anari:` value-range and
// unit-distance annotations and any transfer function authored on the Stage.
bool convertVolume(
    ImportContext &ctx, const pxr::SdfPath &primPath, LayerNodeRef node);

} // namespace tsd::io::usd
