// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/io/importers/detail/usd/UsdImportContext.h"
// usd
#include <pxr/imaging/hd/sceneIndex.h>
// std
#include <string>

namespace tsd::io::usd {

/*
 * A material as TSD sees it, together with the primvar its own texture-reader
 * node asked for. The UV name travels with the material because the geometry
 * converter must bind that primvar rather than assume a conventional name.
 */
struct ResolvedMaterial
{
  MaterialRef material;
  std::string uvPrimvarName;
};

// Resolve the material bound at `materialPath` in the resolved scene, honouring
// the Render Context preference order with per-material fallback. Results are
// cached on the context so a shared material converts once.
ResolvedMaterial resolveMaterial(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &materialPath);

} // namespace tsd::io::usd
