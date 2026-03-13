// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace tsd::core {

// GPU-accelerated vortical field computation.
// Same signature and semantics as vort() in vort.h, but runs on the GPU.
// All pointers are host pointers; data transfer is managed internally.
// Any output pointer may be null; null outputs are skipped.
void vort_cuda(const float *u,
    const float *v,
    const float *w,
    const double *x_,
    const double *y_,
    const double *z_,
    float *vorticity,
    float *helicity,
    float *lambda2,
    float *qCriterion,
    size_t nx,
    size_t ny,
    size_t nz);

// GPU-accelerated Green-Gauss gradient + vorticity for unstructured meshes.
// positions: float3 interleaved (numPoints*3 floats).
// u/v/w: per-vertex scalar velocity components.
// connectivity/cellIndex/cellTypes: ANARI unstructured topology arrays.
// vtxCellOffsets/vtxCellList: vertex-to-cell CSR adjacency (host-built).
// Any output pointer may be null; null outputs are skipped.
void vort_cuda_unstructured(const float *positions,
    const float *u,
    const float *v,
    const float *w,
    const uint32_t *connectivity,
    const uint32_t *cellIndex,
    const uint8_t *cellTypes,
    const uint32_t *vtxCellOffsets,
    const uint32_t *vtxCellList,
    size_t numPoints,
    size_t numCells,
    size_t connSize,
    float *vorticity,
    float *helicity,
    float *lambda2,
    float *qCriterion);

} // namespace tsd::core
