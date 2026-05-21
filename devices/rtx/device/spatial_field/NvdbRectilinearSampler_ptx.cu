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

// OptiX direct-callable entry points for the NanoVDB rectilinear-grid sampler.
// Implementations live in NvdbRectilinearSamplerInline.h.

#include "NvdbRectilinearSamplerInline.h"
#include "gpu/volumeIntegrationDetail.h"

using namespace visrtx;

// Fp4 rectilinear sampler
VISRTX_CALLABLE void __direct_callable__initNvdbRectilinearSamplerFp4(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbRectilinearSampler(samplerState->nvdbRectilinearFp4, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbRectilinearFp4(
    const VolumeSamplingState *samplerState,
    const vec3 *location,
    vec3 *gradient)
{
  return sampleNvdbRectilinear(
      samplerState->nvdbRectilinearFp4, location, gradient);
}

// Fp8 rectilinear sampler
VISRTX_CALLABLE void __direct_callable__initNvdbRectilinearSamplerFp8(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbRectilinearSampler(samplerState->nvdbRectilinearFp8, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbRectilinearFp8(
    const VolumeSamplingState *samplerState,
    const vec3 *location,
    vec3 *gradient)
{
  return sampleNvdbRectilinear(
      samplerState->nvdbRectilinearFp8, location, gradient);
}

// Fp16 rectilinear sampler
VISRTX_CALLABLE void __direct_callable__initNvdbRectilinearSamplerFp16(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbRectilinearSampler(samplerState->nvdbRectilinearFp16, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbRectilinearFp16(
    const VolumeSamplingState *samplerState,
    const vec3 *location,
    vec3 *gradient)
{
  return sampleNvdbRectilinear(
      samplerState->nvdbRectilinearFp16, location, gradient);
}

// FpN rectilinear sampler
VISRTX_CALLABLE void __direct_callable__initNvdbRectilinearSamplerFpN(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbRectilinearSampler(samplerState->nvdbRectilinearFpN, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbRectilinearFpN(
    const VolumeSamplingState *samplerState,
    const vec3 *location,
    vec3 *gradient)
{
  return sampleNvdbRectilinear(
      samplerState->nvdbRectilinearFpN, location, gradient);
}

// Float rectilinear sampler
VISRTX_CALLABLE void __direct_callable__initNvdbRectilinearSamplerFloat(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbRectilinearSampler(samplerState->nvdbRectilinearFloat, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbRectilinearFloat(
    const VolumeSamplingState *samplerState,
    const vec3 *location,
    vec3 *gradient)
{
  return sampleNvdbRectilinear(
      samplerState->nvdbRectilinearFloat, location, gradient);
}

// Woodcock-body callables — see NvdbRegularSampler_ptx.cu for the design rationale.
#define VISRTX_DEFINE_NVDB_RECT_WOODCOCK_CALLABLES(Suffix, ValueType)         \
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
    SamplerStateBox<NvdbRectilinearSamplerState<ValueType>> stateBox;         \
    auto &samplerState = stateBox.state;                                      \
    initNvdbRectilinearSampler(samplerState, &field);                         \
    return detail::woodcockSampleDistance(*ss,                                \
        *hit,                                                                 \
        samplerState,                                                         \
        field,                                                                \
        *albedo,                                                              \
        *extinction,                                                          \
        *didScatter,                                                          \
        normal,                                                               \
        [] __device__(const NvdbRectilinearSamplerState<ValueType> &s,        \
            const SpatialFieldGPUData &,                                      \
            const vec3 &p) { return sampleNvdbRectilinear(s, &p, nullptr); }, \
        [] __device__(const NvdbRectilinearSamplerState<ValueType> &s,        \
            const SpatialFieldGPUData &,                                      \
            const vec3 &p,                                                    \
            vec3 &g) { return sampleNvdbRectilinear(s, &p, &g); });           \
  }                                                                           \
                                                                              \
  VISRTX_CALLABLE void __direct_callable__ratioTrackTransmittance##Suffix(    \
      ScreenSample *ss, const VolumeHit *hit, vec3 *attenuation)              \
  {                                                                           \
    const auto &field = getSpatialFieldData(                                  \
        *ss->frameData, hit->volume->data.tf1d.field);                        \
    SamplerStateBox<NvdbRectilinearSamplerState<ValueType>> stateBox;         \
    auto &samplerState = stateBox.state;                                      \
    initNvdbRectilinearSampler(samplerState, &field);                         \
    detail::woodcockRatioTrackTransmittance(*ss,                              \
        *hit,                                                                 \
        samplerState,                                                         \
        field,                                                                \
        *attenuation,                                                         \
        [] __device__(const NvdbRectilinearSamplerState<ValueType> &s,        \
            const SpatialFieldGPUData &,                                      \
            const vec3 &p) { return sampleNvdbRectilinear(s, &p, nullptr); });\
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
    SamplerStateBox<NvdbRectilinearSamplerState<ValueType>> stateBox;         \
    auto &samplerState = stateBox.state;                                      \
    initNvdbRectilinearSampler(samplerState, &field);                         \
    return detail::latticeRayMarchVolume(*ss,                                 \
        *hit,                                                                 \
        samplerState,                                                         \
        field,                                                                \
        color,                                                                \
        normal,                                                               \
        *opacity,                                                             \
        invSamplingRate,                                                      \
        [] __device__(const NvdbRectilinearSamplerState<ValueType> &s,        \
            const SpatialFieldGPUData &,                                      \
            const vec3 &p) { return sampleNvdbRectilinear(s, &p, nullptr); }, \
        [] __device__(const NvdbRectilinearSamplerState<ValueType> &s,        \
            const SpatialFieldGPUData &,                                      \
            const vec3 &p,                                                    \
            vec3 &g) { return sampleNvdbRectilinear(s, &p, &g); });           \
  }

VISRTX_DEFINE_NVDB_RECT_WOODCOCK_CALLABLES(NvdbRectilinearFp4, nanovdb::Fp4)
VISRTX_DEFINE_NVDB_RECT_WOODCOCK_CALLABLES(NvdbRectilinearFp8, nanovdb::Fp8)
VISRTX_DEFINE_NVDB_RECT_WOODCOCK_CALLABLES(NvdbRectilinearFp16, nanovdb::Fp16)
VISRTX_DEFINE_NVDB_RECT_WOODCOCK_CALLABLES(NvdbRectilinearFpN, nanovdb::FpN)
VISRTX_DEFINE_NVDB_RECT_WOODCOCK_CALLABLES(NvdbRectilinearFloat, float)

#undef VISRTX_DEFINE_NVDB_RECT_WOODCOCK_CALLABLES
