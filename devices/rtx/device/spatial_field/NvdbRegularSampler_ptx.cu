/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// OptiX direct-callable entry points for the NanoVDB regular-grid sampler.
// The actual implementations live in NvdbRegularSamplerInline.h so the volume
// integrator can call them inline on hot paths; this file only registers them
// against the SBT slots for the fallback dispatch path.

#include "NvdbRegularSamplerInline.h"
#include "gpu/volumeIntegrationDetail.h"

using namespace visrtx;

// Fp4 sampler
VISRTX_CALLABLE void __direct_callable__initNvdbSamplerFp4(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbSampler(samplerState->nvdbFp4, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbFp4(
    const VolumeSamplingState *samplerState, const vec3 *location,
    vec3 *gradient)
{
  return sampleNvdb(samplerState->nvdbFp4, location, gradient);
}

// Fp8 sampler
VISRTX_CALLABLE void __direct_callable__initNvdbSamplerFp8(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbSampler(samplerState->nvdbFp8, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbFp8(
    const VolumeSamplingState *samplerState, const vec3 *location,
    vec3 *gradient)
{
  return sampleNvdb(samplerState->nvdbFp8, location, gradient);
}

// Fp16 sampler
VISRTX_CALLABLE void __direct_callable__initNvdbSamplerFp16(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbSampler(samplerState->nvdbFp16, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbFp16(
    const VolumeSamplingState *samplerState, const vec3 *location,
    vec3 *gradient)
{
  return sampleNvdb(samplerState->nvdbFp16, location, gradient);
}

// FpN sampler
VISRTX_CALLABLE void __direct_callable__initNvdbSamplerFpN(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbSampler(samplerState->nvdbFpN, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbFpN(
    const VolumeSamplingState *samplerState, const vec3 *location,
    vec3 *gradient)
{
  return sampleNvdb(samplerState->nvdbFpN, location, gradient);
}

// Float sampler
VISRTX_CALLABLE void __direct_callable__initNvdbSamplerFloat(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbSampler(samplerState->nvdbFloat, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbFloat(
    const VolumeSamplingState *samplerState, const vec3 *location,
    vec3 *gradient)
{
  return sampleNvdb(samplerState->nvdbFloat, location, gradient);
}

// Woodcock-body callables — one per variant. Each stack-allocates the typed
// sampler state, inits it once, then runs the shared body from
// volumeIntegrationDetail.h with __device__ lambdas resolving to the
// variant's inline sampleNvdb<T>.
#define VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES(Suffix, ValueType)              \
  VISRTX_CALLABLE float __direct_callable__sampleDistance##Suffix(            \
      ScreenSample *ss,                                                       \
      const VolumeHit *hit,                                                   \
      vec3 *albedo,                                                           \
      float *extinction,                                                      \
      bool *didScatter,                                                       \
      vec3 *normal)                                                           \
  {                                                                           \
    const auto &field = getSpatialFieldData(                                  \
        *ss->frameData, hit->volume->data.tf1d.field);                        \
    SamplerStateBox<NvdbRegularSamplerState<ValueType>> stateBox;             \
    auto &samplerState = stateBox.state;                                      \
    initNvdbSampler(samplerState, &field);                                    \
    return detail::woodcockSampleDistance(*ss,                                \
        *hit,                                                                 \
        samplerState,                                                         \
        field,                                                                \
        *albedo,                                                              \
        *extinction,                                                          \
        *didScatter,                                                          \
        normal,                                                               \
        [] __device__(const NvdbRegularSamplerState<ValueType> &s,            \
            const SpatialFieldGPUData &,                                      \
            const vec3 &p) { return sampleNvdb(s, &p, nullptr); },            \
        [] __device__(const NvdbRegularSamplerState<ValueType> &s,            \
            const SpatialFieldGPUData &,                                      \
            const vec3 &p,                                                    \
            vec3 &g) { return sampleNvdb(s, &p, &g); });                      \
  }                                                                           \
                                                                              \
  VISRTX_CALLABLE void __direct_callable__ratioTrackTransmittance##Suffix(    \
      ScreenSample *ss, const VolumeHit *hit, vec3 *attenuation)              \
  {                                                                           \
    const auto &field = getSpatialFieldData(                                  \
        *ss->frameData, hit->volume->data.tf1d.field);                        \
    SamplerStateBox<NvdbRegularSamplerState<ValueType>> stateBox;             \
    auto &samplerState = stateBox.state;                                      \
    initNvdbSampler(samplerState, &field);                                    \
    detail::woodcockRatioTrackTransmittance(*ss,                              \
        *hit,                                                                 \
        samplerState,                                                         \
        field,                                                                \
        *attenuation,                                                         \
        [] __device__(const NvdbRegularSamplerState<ValueType> &s,            \
            const SpatialFieldGPUData &,                                      \
            const vec3 &p) { return sampleNvdb(s, &p, nullptr); });           \
  }                                                                           \
                                                                              \
  VISRTX_CALLABLE float __direct_callable__rayMarchVolume##Suffix(            \
      ScreenSample *ss,                                                       \
      const VolumeHit *hit,                                                   \
      vec3 *color,                                                            \
      vec3 *normal,                                                           \
      float *opacity,                                                         \
      float invSamplingRate)                                                  \
  {                                                                           \
    const auto &field = getSpatialFieldData(                                  \
        *ss->frameData, hit->volume->data.tf1d.field);                        \
    SamplerStateBox<NvdbRegularSamplerState<ValueType>> stateBox;             \
    auto &samplerState = stateBox.state;                                      \
    initNvdbSampler(samplerState, &field);                                    \
    return detail::latticeRayMarchVolume(*ss,                                 \
        *hit,                                                                 \
        samplerState,                                                         \
        field,                                                                \
        color,                                                                \
        normal,                                                               \
        *opacity,                                                             \
        invSamplingRate,                                                      \
        [] __device__(const NvdbRegularSamplerState<ValueType> &s,            \
            const SpatialFieldGPUData &,                                      \
            const vec3 &p) { return sampleNvdb(s, &p, nullptr); },            \
        [] __device__(const NvdbRegularSamplerState<ValueType> &s,            \
            const SpatialFieldGPUData &,                                      \
            const vec3 &p,                                                    \
            vec3 &g) { return sampleNvdb(s, &p, &g); });                      \
  }

VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES(NvdbFp4, nanovdb::Fp4)
VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES(NvdbFp8, nanovdb::Fp8)
VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES(NvdbFp16, nanovdb::Fp16)
VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES(NvdbFpN, nanovdb::FpN)
VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES(NvdbFloat, float)

#undef VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES
