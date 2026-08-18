// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/io/importers/detail/usd/UsdImportContext.h"
#include "tsd/io/usd/UsdResolvedGeometry.h"
// usd
#include <pxr/imaging/hd/sceneIndex.h>
// std
#include <string>
#include <utility>
#include <vector>

namespace tsd::io::usd {

/*
 * What an Import made of one gprim: the Surfaces it emitted, and everything a
 * later resolve of the same prim needs in order to reproduce exactly this
 * conversion. An animation binding keeps the second half so that a scrub
 * re-fills these Geometries rather than building new ones (ADR 0022).
 */
struct ConvertedGeometry
{
  std::vector<SurfaceRef> surfaces;

  // The Geometry each resolved Part became, by Part name. Names are the prim's
  // own path and its subsets' paths, so they survive a re-resolve.
  std::vector<std::pair<std::string, GeometryRef>> geometryByPart;

  // Reproduces the attribute-slot assignment and baking this conversion used.
  // Replaying it is what keeps a scrub from having to resolve materials again.
  GeometryResolveOptions resolveOptions;
};

// Convert one resolved gprim into TSD Surfaces. A mesh carrying per-face
// material subsets yields several Surfaces, one per subset, sharing the mesh's
// vertex arrays. `bakeXform` is baked into the emitted vertex data and is the
// identity for everything but Prototype-internal geometry (ADR 0016).
ConvertedGeometry convertGeometry(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const tsd::math::mat4 &bakeXform);

} // namespace tsd::io::usd
