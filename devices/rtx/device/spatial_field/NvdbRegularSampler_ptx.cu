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
// Only the Woodcock-body callables are exposed via the SBT; value/normal/init
// sampling stays inline (NvdbRegularSamplerInline.h) since the bodies below
// resolve sampleValue/sampleNormal by ADL on the concrete state type, and the
// volume integrator calls the inline sampleValue/sampleNormal directly on hot
// paths.

#include "NvdbRegularSamplerInline.h"
#include "gpu/volumeIntegrationDetail.h"

using namespace visrtx;

// Woodcock-body callables — one per variant. Each stack-allocates the typed
// sampler state, inits it once, then runs the shared body from
// volumeIntegrationDetail.h, which calls the variant's inline
// sampleValue/sampleNormal.
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
        normal);                                                              \
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
    detail::woodcockRatioTrackTransmittance(                                  \
        *ss, *hit, samplerState, field, *attenuation);                        \
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
        invSamplingRate);                                                     \
  }

VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES(NvdbFp4, nanovdb::Fp4)
VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES(NvdbFp8, nanovdb::Fp8)
VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES(NvdbFp16, nanovdb::Fp16)
VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES(NvdbFpN, nanovdb::FpN)
VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES(NvdbFloat, float)

#undef VISRTX_DEFINE_NVDB_WOODCOCK_CALLABLES
