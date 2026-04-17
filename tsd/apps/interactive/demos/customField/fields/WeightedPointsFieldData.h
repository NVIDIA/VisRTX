// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu/gpu_math.h"
#include <cstdint>

namespace visrtx {
constexpr uint32_t DEMO_WEIGHTED_POINTS_FIELD_TYPE = 201;
} // namespace visrtx

struct WeightedPointsFieldData
{
  const float *values;       // device ptr: 4 floats per node (x,y,z,value)
  const int32_t *indices;    // device ptr: 2 ints per node (childBegin,childEnd)
  int32_t numNodes;
  float sigma;               // Gaussian width
  float inv2SigmaSq;         // 1 / (2 * sigma^2), precomputed
  float cutoff;              // initial LOD distance
  visrtx::vec3 domainMin;
  visrtx::vec3 domainMax;
};

static_assert(sizeof(WeightedPointsFieldData) <= 256,
    "WeightedPointsFieldData exceeds CustomData::fieldData size");
