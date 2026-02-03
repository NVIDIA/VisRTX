/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "SpatialField.h"

namespace visrtx {

/**
 * @brief Abstract base class for analytical (procedural) spatial fields
 *
 * Analytical fields compute their values procedurally rather than from
 * stored data. They provide a framework for implementing custom fields.
 *
 * Subclasses must implement:
 * - commitParameters(): Parse ANARI parameters
 * - finalize(): Prepare GPU data
 * - bounds(): Return field bounding box
 * - stepSize(): Return ray marching step size
 */
struct AnalyticalField : public SpatialField
{
  AnalyticalField(DeviceGlobalState *d) : SpatialField(d) {}
  ~AnalyticalField() override = default;
};

} // namespace visrtx
