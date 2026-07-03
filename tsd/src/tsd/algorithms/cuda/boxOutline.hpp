// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/TSDMath.hpp"
// cuda
#include <cuda_runtime_api.h>
// std
#include <cstdint>

namespace tsd::algorithms::cuda {

void boxOutline(cudaStream_t stream,
    const tsd::math::box3 &box,
    const tsd::math::mat4 &projView,
    const tsd::math::float3 &eye,
    const tsd::math::float3 &dir,
    bool orthographicDepth,
    const float *depth,
    uint32_t *color,
    uint32_t outlineColor,
    uint32_t lineWidth,
    uint32_t width,
    uint32_t height);

void boxOutline(const tsd::math::box3 &box,
    const tsd::math::mat4 &projView,
    const tsd::math::float3 &eye,
    const tsd::math::float3 &dir,
    bool orthographicDepth,
    const float *depth,
    uint32_t *color,
    uint32_t outlineColor,
    uint32_t lineWidth,
    uint32_t width,
    uint32_t height);

} // namespace tsd::algorithms::cuda
