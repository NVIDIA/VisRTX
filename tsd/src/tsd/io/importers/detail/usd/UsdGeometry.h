// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/io/importers/detail/usd/UsdImportContext.h"
// usd
#include <pxr/imaging/hd/sceneIndex.h>
// std
#include <vector>

namespace tsd::io::usd {

// True for the resolved prim types this converter turns into TSD geometry.
bool isGeometryPrimType(const pxr::TfToken &primType);

// Convert one resolved gprim into TSD Surfaces. A mesh carrying per-face
// material subsets yields several Surfaces, one per subset, sharing the mesh's
// vertex arrays. `bakeXform` is baked into the emitted vertex data and is the
// identity for everything but Prototype-internal geometry (ADR 0016).
std::vector<SurfaceRef> convertGeometry(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const tsd::math::mat4 &bakeXform);

} // namespace tsd::io::usd
