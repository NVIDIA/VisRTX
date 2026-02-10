/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "SpatialField.h"

namespace visrtx {

/**
 * @brief Abstract base class for procedural spatial fields
 *
 * Custom fields compute their values procedurally rather than from
 * stored data. They provide a framework for implementing custom fields.
 *
 * Subclasses must implement:
 * - commitParameters(): Parse ANARI parameters
 * - finalize(): Prepare GPU data
 * - bounds(): Return field bounding box
 * - stepSize(): Return ray marching step size
 */
struct CustomField : public SpatialField
{
  CustomField(DeviceGlobalState *d) : SpatialField(d) {}
  ~CustomField() override = default;
};

} // namespace visrtx
