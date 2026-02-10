// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "optix_visrtx.h"

namespace visrtx {

/**
 * @brief PTX wrapper for custom field samplers
 *
 * This single module contains callable programs for all custom field
 * subtypes
 */
struct CustomFieldSampler
{
  static ptx_blob ptx();
};

} // namespace visrtx
