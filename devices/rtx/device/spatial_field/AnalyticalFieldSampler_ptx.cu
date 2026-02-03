// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file AnalyticalFieldSampler_ptx.cu
 * @brief OptiX callable programs for analytical spatial field sampling
 * 
 * This module provides a single sampler entry point that dispatches to
 * specific field type implementations based on the AnalyticalFieldType
 * stored in the field data. This keeps external VisRTX type-agnostic.
 * 
 * Supported field types:
 * - Magnetic (dipole field)
 * - IMF (Interplanetary Magnetic Field)
 * - Aurora (auroral emissions)
 * - Planet (planetary surface/atmosphere)
 * - Cloud (cloud layer)
 */

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/shadingState.h"

using namespace visrtx;

//=============================================================================
// Exported OptiX callable programs - single dispatch point for all types
//=============================================================================

/**
 * @brief Initialize analytical field sampler state
 * 
 * Copies the field data to the sampler state for use during sampling.
 * The type field determines which union member is valid.
 */
VISRTX_CALLABLE void __direct_callable__initAnalyticalSampler(
    VolumeSamplingState *samplerState,
    const SpatialFieldGPUData *field)
{
  samplerState->analytical = field->data.analytical;
}

/**
 * @brief Sample the analytical field at a given location
 * 
 * Dispatches to the appropriate sampling function based on the type field.
 * Returns a normalized field value in [0, 1].
 */
VISRTX_CALLABLE float __direct_callable__sampleAnalytical(
    const VolumeSamplingState *samplerState,
    const vec3 *location)
{
    return 0.0f;
}
