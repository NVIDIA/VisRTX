// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file CustomFieldSampler_ptx.cu
 * @brief OptiX callable programs for custom spatial field sampling
 *
 * This file provides the OptiX callable entry points for custom fields.
 * The actual sampling implementations are provided by including external
 * sampler headers that define per-field-type sampling functions.
 *
 * To add a new custom field type:
 * 1. Define the field data struct and add to CustomFieldType enum
 * 2. Create a sampler header with sampleXxx() function
 * 3. Include the header below
 * 4. Add a case to the switch in __direct_callable__sampleCustom
 */

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/shadingState.h"

// Include custom field data definitions (provides CustomFieldType enum
// and field-specific data structures)
#ifdef VISRTX_CUSTOM_FIELD_DATA_HEADER
#include VISRTX_CUSTOM_FIELD_DATA_HEADER
#endif

// Include per-field sampler implementations
#ifdef VISRTX_CUSTOM_SAMPLERS_HEADER
#include VISRTX_CUSTOM_SAMPLERS_HEADER
#endif

using namespace visrtx;

//=============================================================================
// Exported OptiX callable programs
//=============================================================================

/**
 * @brief Initialize custom field sampler state
 *
 * Copies the field data to the sampler state for use during sampling.
 */
VISRTX_CALLABLE void __direct_callable__initCustomSampler(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  samplerState->custom = field->data.custom;
}

/**
 * @brief Sample the custom field at a given location
 *
 * Dispatches to the appropriate sampling function based on subType.
 * Returns a normalized field value in [0, 1].
 *
 * If no custom samplers are configured (VISRTX_CUSTOM_SAMPLE_DISPATCH
 * not defined), returns 0.0 as a fallback.
 */
VISRTX_CALLABLE float __direct_callable__sampleCustom(
    const VolumeSamplingState *samplerState, const vec3 *location)
{
#ifdef VISRTX_CUSTOM_SAMPLE_DISPATCH
  const CustomFieldData &data = samplerState->custom;
  const vec3 P = *location;

  // Dispatch macro expands to switch statement with all registered field types
  VISRTX_CUSTOM_SAMPLE_DISPATCH(data, P)
#else
  // No custom field types configured - return default value
  return 0.0f;
#endif
}
