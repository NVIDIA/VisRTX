// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "WeightedPointsFieldData.h"
#include "samplers/ValueRangeWeightedPoints.cuh"
#include "samplers/SampleWeightedPoints.cuh"

#define VISRTX_CUSTOM_SAMPLE_DISPATCH(data, P) \
  switch (data.subType) {                          \
    case visrtx::DEMO_WEIGHTED_POINTS_FIELD_TYPE:  \
      return sampleWeightedPoints(                 \
          *reinterpret_cast<const WeightedPointsFieldData*>(data.fieldData), P); \
    default:                                       \
      return 0.0f;                                 \
  }

#define VISRTX_CUSTOM_VALUE_RANGE_DISPATCH(data, boxLo, boxHi)                 \
  switch (data.subType) {                                                      \
    case visrtx::DEMO_WEIGHTED_POINTS_FIELD_TYPE:                              \
      return valueRangeWeightedPoints(                                         \
          *reinterpret_cast<const WeightedPointsFieldData*>(data.fieldData),   \
          boxLo, boxHi);                                                       \
    default: /* unknown subtype: global conservative interval, never vanish */ \
      return visrtx::box1{0.f,                                                  \
          reinterpret_cast<const WeightedPointsFieldData*>(                    \
              data.fieldData)->maxValue};                                      \
  }
