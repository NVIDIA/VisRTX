/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// OptiX direct-callable entry points for the structured rectilinear sampler.
// Implementations live in StructuredRectilinearSamplerInline.h.

#include "StructuredRectilinearSamplerInline.h"
#include "gpu/volumeIntegrationDetail.h"

using namespace visrtx;

VISRTX_CALLABLE void __direct_callable__initStructuredRectilinearSampler(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initStructuredRectilinearSampler(
      samplerState->structuredRectilinear, field);
}

VISRTX_CALLABLE float __direct_callable__sampleStructuredRectilinear(
    const VolumeSamplingState *samplerState,
    const vec3 *location,
    vec3 *gradient)
{
  return sampleStructuredRectilinear(
      samplerState->structuredRectilinear, location, gradient);
}

VISRTX_CALLABLE float
__direct_callable__sampleDistanceStructuredRectilinear(ScreenSample *ss,
    const VolumeHit *hit,
    vec3 *albedo,
    float *extinction,
    bool *didScatter,
    vec3 *normal)
{
  const auto &field =
      getSpatialFieldData(*ss->frameData, hit->volume->data.tf1d.field);
  SamplerStateBox<StructuredRectilinearSamplerState> stateBox;
  auto &samplerState = stateBox.state;
  initStructuredRectilinearSampler(samplerState, &field);
  return detail::woodcockSampleDistance(*ss,
      *hit,
      samplerState,
      field,
      *albedo,
      *extinction,
      *didScatter,
      normal,
      [] __device__(const StructuredRectilinearSamplerState &s,
          const SpatialFieldGPUData &,
          const vec3 &p) {
        return sampleStructuredRectilinear(s, &p, nullptr);
      },
      [] __device__(const StructuredRectilinearSamplerState &s,
          const SpatialFieldGPUData &,
          const vec3 &p,
          vec3 &g) { return sampleStructuredRectilinear(s, &p, &g); });
}

VISRTX_CALLABLE void
__direct_callable__ratioTrackTransmittanceStructuredRectilinear(
    ScreenSample *ss, const VolumeHit *hit, vec3 *attenuation)
{
  const auto &field =
      getSpatialFieldData(*ss->frameData, hit->volume->data.tf1d.field);
  SamplerStateBox<StructuredRectilinearSamplerState> stateBox;
  auto &samplerState = stateBox.state;
  initStructuredRectilinearSampler(samplerState, &field);
  detail::woodcockRatioTrackTransmittance(*ss,
      *hit,
      samplerState,
      field,
      *attenuation,
      [] __device__(const StructuredRectilinearSamplerState &s,
          const SpatialFieldGPUData &,
          const vec3 &p) {
        return sampleStructuredRectilinear(s, &p, nullptr);
      });
}

VISRTX_CALLABLE float __direct_callable__rayMarchVolumeStructuredRectilinear(
    ScreenSample *ss,
    const VolumeHit *hit,
    vec3 *color,
    vec3 *normal,
    float *opacity,
    float invSamplingRate)
{
  const auto &field =
      getSpatialFieldData(*ss->frameData, hit->volume->data.tf1d.field);
  SamplerStateBox<StructuredRectilinearSamplerState> stateBox;
  auto &samplerState = stateBox.state;
  initStructuredRectilinearSampler(samplerState, &field);
  return detail::latticeRayMarchVolume(*ss,
      *hit,
      samplerState,
      field,
      color,
      normal,
      *opacity,
      invSamplingRate,
      [] __device__(const StructuredRectilinearSamplerState &s,
          const SpatialFieldGPUData &,
          const vec3 &p) {
        return sampleStructuredRectilinear(s, &p, nullptr);
      },
      [] __device__(const StructuredRectilinearSamplerState &s,
          const SpatialFieldGPUData &,
          const vec3 &p,
          vec3 &g) { return sampleStructuredRectilinear(s, &p, &g); });
}
