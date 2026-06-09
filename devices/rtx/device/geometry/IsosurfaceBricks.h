/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "gpu/gpu_objects.h"
#include "utility/DeviceBuffer.h"
// cuda
#include <cuda_runtime.h>

namespace visrtx {

// Builds a coarse "active-region" BLAS for an isosurface: the field's macrocell
// grid is tiled into bricks (BRICK_SIZE macrocells per edge); each brick that
// contains a macrocell whose value range brackets an isovalue is emitted as one
// AABB tightly bounding its active macrocells. This lets the BVH cull rays that
// miss the active region while keeping the primitive count far below one box
// per macrocell; the intersection program's DDA does the fine per-cell skipping
// inside each brick. Returns the number of active bricks (compacted into
// outAabbs as box3). Returns 0 (and writes nothing) for an empty active set.
size_t buildIsosurfaceBricks(cudaStream_t stream,
    const UniformGridData &grid,
    const float *isovaluesDev,
    uint32_t numIsovalues,
    DeviceBuffer &outAabbs);

} // namespace visrtx
