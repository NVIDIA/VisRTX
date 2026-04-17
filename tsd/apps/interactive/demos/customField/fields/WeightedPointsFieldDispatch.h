// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "WeightedPointsFieldData.h"
#include "samplers/SampleWeightedPoints.cuh"

#define VISRTX_CUSTOM_SAMPLE_DISPATCH(data, P) \
  switch (data.subType) {                          \
    case visrtx::DEMO_WEIGHTED_POINTS_FIELD_TYPE:  \
      return sampleWeightedPoints(                 \
          *reinterpret_cast<const WeightedPointsFieldData*>(data.fieldData), P); \
    default:                                       \
      return 0.0f;                                 \
  }
