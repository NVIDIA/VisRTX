// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/io/importers/detail/usd/UsdImportContext.h"
// usd
#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/path.h>
// std
#include <string>

namespace tsd::io::usd {

// Whether the resolved scene's prim type is a UsdVol Volume.
bool isVolumePrimType(const pxr::TfToken &primType);

// Import a UsdVol Volume prim, honouring the `anari:` value-range and
// unit-distance annotations and any transfer function authored on the Stage.
// Returns false for a Volume whose field TSD could not load, which the caller
// reports rather than dropping silently; `skipDetail` carries back which of
// the ways to have no field this Volume took.
bool convertVolume(ImportContext &ctx,
    const pxr::SdfPath &primPath,
    LayerNodeRef node,
    std::string *skipDetail);

} // namespace tsd::io::usd
