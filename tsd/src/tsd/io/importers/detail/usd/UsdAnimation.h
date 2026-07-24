// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/io/importers/detail/usd/UsdImportContext.h"

namespace tsd::io::usd {

// Bind a node's transform to the prim's authored sample times, decomposed into
// rotation, translation, and scale. Extra samples are inserted only across
// intervals whose rotation a two-key spherical interpolation would collapse.
void addTransformAnimation(
    ImportContext &ctx, const pxr::SdfPath &primPath, LayerNodeRef node);

// Bind a geometry's vertex arrays to the retained Stage so that a long
// animation of a dense mesh is pulled on demand instead of held in memory
// (ADR 0018). Does nothing unless the prim's points are time-sampled.
void addDeformingGeometryAnimation(
    ImportContext &ctx, const pxr::SdfPath &primPath, GeometryRef geometry);

} // namespace tsd::io::usd
