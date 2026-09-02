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

#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define VISRTX_HOST_DEVICE inline __host__ __device__
#define VISRTX_FORCE_INLINE __forceinline__

#define VISRTX_GLOBAL extern "C" __global__
#define VISRTX_DEVICE inline __device__
#define VISRTX_CALLABLE extern "C" __device__

#else
#define VISRTX_HOST_DEVICE inline
#define VISRTX_GLOBAL extern "C"
#define VISRTX_DEVICE inline
#define VISRTX_CALLABLE extern "C"
#endif

#define NUM_SBT_PRIMITIVE_INTERSECTOR_ENTRIES 4
#define SBT_TRIANGLE_OFFSET 0
#define SBT_CURVE_OFFSET 1
#define SBT_CUSTOM_OFFSET 2
// Analytic area-light proxies (ADR 0009). A dedicated slot, not a case inside
// the custom intersector, so proxies get their own closest-hit program: the
// shared one dereferences a hit's material/geometry/surface-instance records
// and a proxy has none of the three.
#define SBT_LIGHT_PROXY_OFFSET 3

// OptiX instance visibility masks (ADR 0009).
//
// These make "which rays can see a light proxy" a property of the trace call
// rather than a branch in every renderer. A renderer that does not handle proxy
// hits simply never sets VISRTX_MASK_LIGHT_PROXY_*, so its rays cannot reach a
// proxy at all — pass-through is structural, and costs nothing.
//
// Shadow rays never set either proxy bit, so a light can neither shadow the
// scene nor shadow itself. This is a mask exclusion rather than a distance trim
// because the proxy is not real occluding geometry.
constexpr uint32_t VISRTX_MASK_GEOMETRY = 1u;
// Proxy of a light with visible=true: reachable by camera rays.
constexpr uint32_t VISRTX_MASK_LIGHT_PROXY_VISIBLE = 2u;
// Proxy of a light with visible=false: hidden from camera rays, but still
// reachable by reflection/GI rays so indirect illumination stays consistent.
constexpr uint32_t VISRTX_MASK_LIGHT_PROXY_HIDDEN = 4u;
