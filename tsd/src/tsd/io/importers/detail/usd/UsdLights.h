// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/io/importers/detail/usd/UsdImportContext.h"
// usd
#include <pxr/imaging/hd/sceneIndex.h>

namespace tsd::io::usd {

bool isLightPrimType(const pxr::TfToken &primType);

// Convert one resolved light prim. Returns a null ref for light types TSD
// cannot represent, which the caller reports rather than dropping silently.
LightRef convertLight(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim);

// Cameras import into the Scene's camera pool with their world transform,
// animated through bindings where a rig authors motion anywhere above them.
void convertCamera(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim);

} // namespace tsd::io::usd
