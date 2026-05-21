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
#include "gpu/sbt.h"
#include "gpu/shadingState.h"
#include "gpu/volumeIntegrationDetail.h"

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
    const VolumeSamplingState *samplerState,
    const vec3 *location,
    vec3 *gradient)
{
#ifdef VISRTX_CUSTOM_SAMPLE_DISPATCH
  const CustomFieldData &data = samplerState->custom;
  const vec3 P = *location;

  VISRTX_CUSTOM_SAMPLE_DISPATCH(data, P)
#else
  return 0.0f;
#endif
}

//=============================================================================
// Woodcock-body callables for custom fields. Because the inner sample()
// implementation is user-supplied (compiled into __direct_callable__sampleCustom
// in this same module), the Woodcock body dispatches via optixDirectCall to the
// same module's Sample slot — callable-in-callable. Slower than the built-in
// families' inline path but preserves the user-facing UX (custom-field authors
// implement only init + sample).
//=============================================================================

VISRTX_CALLABLE float __direct_callable__sampleDistanceCustom(ScreenSample *ss,
    const VolumeHit *hit,
    vec3 *albedo,
    float *extinction,
    bool *didScatter,
    vec3 *normal)
{
  const auto &field =
      getSpatialFieldData(*ss->frameData, hit->volume->data.tf1d.field);
  VolumeSamplingState samplerState;
  samplerState.custom = field.data.custom;

  const uint32_t baseIdx = uint32_t(field.samplerCallableIndex);
  return detail::woodcockSampleDistance(*ss,
      *hit,
      samplerState,
      field,
      *albedo,
      *extinction,
      *didScatter,
      normal,
      [baseIdx] __device__(const VolumeSamplingState &s,
          const SpatialFieldGPUData &,
          const vec3 &p) {
        return optixDirectCall<float>(
            baseIdx + uint32_t(SpatialFieldSamplerEntryPoints::Sample),
            &s,
            &p,
            (vec3 *)nullptr);
      },
      [baseIdx] __device__(const VolumeSamplingState &s,
          const SpatialFieldGPUData &,
          const vec3 &p,
          vec3 &g) {
        return optixDirectCall<float>(
            baseIdx + uint32_t(SpatialFieldSamplerEntryPoints::Sample),
            &s,
            &p,
            &g);
      });
}

VISRTX_CALLABLE void __direct_callable__ratioTrackTransmittanceCustom(
    ScreenSample *ss, const VolumeHit *hit, vec3 *attenuation)
{
  const auto &field =
      getSpatialFieldData(*ss->frameData, hit->volume->data.tf1d.field);
  VolumeSamplingState samplerState;
  samplerState.custom = field.data.custom;

  const uint32_t baseIdx = uint32_t(field.samplerCallableIndex);
  detail::woodcockRatioTrackTransmittance(*ss,
      *hit,
      samplerState,
      field,
      *attenuation,
      [baseIdx] __device__(const VolumeSamplingState &s,
          const SpatialFieldGPUData &,
          const vec3 &p) {
        return optixDirectCall<float>(
            baseIdx + uint32_t(SpatialFieldSamplerEntryPoints::Sample),
            &s,
            &p,
            (vec3 *)nullptr);
      });
}

VISRTX_CALLABLE float __direct_callable__rayMarchVolumeCustom(ScreenSample *ss,
    const VolumeHit *hit,
    vec3 *color,
    vec3 *normal,
    float *opacity,
    float invSamplingRate)
{
  const auto &field =
      getSpatialFieldData(*ss->frameData, hit->volume->data.tf1d.field);
  VolumeSamplingState samplerState;
  samplerState.custom = field.data.custom;

  const uint32_t baseIdx = uint32_t(field.samplerCallableIndex);
  return detail::latticeRayMarchVolume(*ss,
      *hit,
      samplerState,
      field,
      color,
      normal,
      *opacity,
      invSamplingRate,
      [baseIdx] __device__(const VolumeSamplingState &s,
          const SpatialFieldGPUData &,
          const vec3 &p) {
        return optixDirectCall<float>(
            baseIdx + uint32_t(SpatialFieldSamplerEntryPoints::Sample),
            &s,
            &p,
            (vec3 *)nullptr);
      },
      [baseIdx] __device__(const VolumeSamplingState &s,
          const SpatialFieldGPUData &,
          const vec3 &p,
          vec3 &g) {
        return optixDirectCall<float>(
            baseIdx + uint32_t(SpatialFieldSamplerEntryPoints::Sample),
            &s,
            &p,
            &g);
      });
}
