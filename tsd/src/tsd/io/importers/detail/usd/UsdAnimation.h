// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/io/importers/detail/usd/UsdGeometry.h"
#include "tsd/io/importers/detail/usd/UsdImportContext.h"
#include "tsd/scene/objects/Array.hpp"
// std
#include <vector>

namespace tsd::io::usd {

// Rescale authored time codes onto the Stage's own time-code range. This is
// the one clock every binding from a single import shares; the relative
// spacing of the authored samples is preserved, so nothing is resampled.
std::vector<float> normalizeSampleTimes(
    const pxr::UsdStageRefPtr &stage, const std::vector<double> &times);

// Bind a node's transform to the prim's authored sample times, decomposed into
// rotation, translation, and scale. Extra samples are inserted only across
// intervals whose rotation a two-key spherical interpolation would collapse.
void addTransformAnimation(
    ImportContext &ctx, const pxr::SdfPath &primPath, LayerNodeRef node);

// How many time samples the placements of a point instancer are authored
// with, taken across every attribute that moves them. Zero or one means the
// instancer does not animate.
size_t pointInstancerSampleCount(
    ImportContext &ctx, const pxr::SdfPath &primPath);

// Bind one Prototype's transform Array to the Stage Session, so that scrubbing
// re-fills the matrices the Import just wrote rather than re-running
// conversion. `arrayNode` is the transform-array node holding `transforms`,
// which the binding needs in order to re-point the node if the placement count
// changes mid-sequence.
void addInstancerAnimation(ImportContext &ctx,
    const pxr::SdfPath &primPath,
    size_t prototypeIndex,
    LayerNodeRef arrayNode,
    ArrayRef transforms);

// Bind everything one converted gprim produced to the Stage Session, so that a
// long animation of a dense mesh is pulled on demand instead of held in memory
// (ADR 0018) and arrives as one consistent set (ADR 0022). Does nothing unless
// the prim's points are time-sampled.
void addDeformingGeometryAnimation(ImportContext &ctx,
    const pxr::SdfPath &primPath,
    ConvertedGeometry &converted);

} // namespace tsd::io::usd
