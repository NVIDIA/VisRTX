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
 * 4. Add a case to the dispatch used by sampleCustomImpl
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

// User value sampler. Returns 0 when no custom samplers are configured. The
// user API exposes only a scalar sampler (VISRTX_CUSTOM_SAMPLE_DISPATCH); the
// normal is derived from it by central differences below.
namespace visrtx {
VISRTX_DEVICE float sampleCustomImpl(const CustomFieldData &data, const vec3 &P)
{
#ifdef VISRTX_CUSTOM_SAMPLE_DISPATCH
  VISRTX_CUSTOM_SAMPLE_DISPATCH(data, P)
#else
  return 0.0f;
#endif
}
} // namespace visrtx

VISRTX_CALLABLE float __direct_callable__sampleValueCustom(
    const VolumeSamplingState *samplerState,
    const SpatialFieldGPUData *,
    const vec3 *location)
{
  return sampleCustomImpl(samplerState->custom, *location);
}

// Central-difference gradient of the user value sampler. Returns the
// unnormalized object-space gradient, matching the built-in sampleNormal
// convention (caller orients + normalizes, see computeWorldNormal). Step is a
// small fraction of the field's domain so it is scale-invariant.
VISRTX_CALLABLE vec3 __direct_callable__sampleNormalCustom(
    const VolumeSamplingState *samplerState,
    const SpatialFieldGPUData *field,
    const vec3 *location)
{
  const CustomFieldData &d = samplerState->custom;
  const vec3 p = *location;
  const vec3 extent = field->roi.upper - field->roi.lower;
  const float eps = fmaxf(1e-3f * glm::length(extent), 1e-6f);
  const float gx = sampleCustomImpl(d, p + vec3(eps, 0.f, 0.f))
      - sampleCustomImpl(d, p - vec3(eps, 0.f, 0.f));
  const float gy = sampleCustomImpl(d, p + vec3(0.f, eps, 0.f))
      - sampleCustomImpl(d, p - vec3(0.f, eps, 0.f));
  const float gz = sampleCustomImpl(d, p + vec3(0.f, 0.f, eps))
      - sampleCustomImpl(d, p - vec3(0.f, 0.f, eps));
  return vec3(gx, gy, gz) * (1.f / (2.f * eps));
}

//=============================================================================
// Woodcock-body callables for custom fields. Because the inner sample()
// implementation is user-supplied (compiled into the value/normal callables in
// this same module), the Woodcock body dispatches via optixDirectCall to the
// same module's SampleValue/SampleNormal slots — callable-in-callable. Slower
// than the built-in families' inline path but preserves the user-facing UX
// (custom-field authors implement only init + sample).
//=============================================================================

// Shared per-sample API (see gpu/volumeIntegrationDetail.h) for custom fields.
// Routes init/value/normal back through the SBT (callable-in-callable); shared
// with the isosurface intersector via CustomFieldSamplerInline.h.
#include "CustomFieldSamplerInline.h"

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
  initSamplerState(samplerState, field);

  return detail::woodcockSampleDistance(*ss,
      *hit,
      samplerState,
      field,
      *albedo,
      *extinction,
      *didScatter,
      normal);
}

VISRTX_CALLABLE void __direct_callable__ratioTrackTransmittanceCustom(
    ScreenSample *ss, const VolumeHit *hit, vec3 *attenuation)
{
  const auto &field =
      getSpatialFieldData(*ss->frameData, hit->volume->data.tf1d.field);
  VolumeSamplingState samplerState;
  initSamplerState(samplerState, field);

  detail::woodcockRatioTrackTransmittance(
      *ss, *hit, samplerState, field, *attenuation);
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
  initSamplerState(samplerState, field);

  return detail::latticeRayMarchVolume(
      *ss, *hit, samplerState, field, color, normal, *opacity, invSamplingRate);
}
