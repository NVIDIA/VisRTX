// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/io/importers/detail/usd/UsdImportContext.h"
// usd
#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/path.h>

namespace tsd::io::usd {

// Whether the resolved scene's prim type is a UsdVol Volume.
bool isVolumePrimType(const pxr::TfToken &primType);

// Import a UsdVol Volume prim, honouring the `anari:` value-range and
// unit-distance annotations and any transfer function authored on the Stage.
bool convertVolume(
    ImportContext &ctx, const pxr::SdfPath &primPath, LayerNodeRef node);

} // namespace tsd::io::usd
