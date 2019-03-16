/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "Common.h"


static __host__ __device__ __inline__ optix::uint packNormal(const optix::float3& normal)
{
    const optix::float3 N = normal * 0.5f + 0.5f;

    const optix::uint packed = ((optix::uint) (N.x * 1023)) |
                    ((optix::uint) (N.y * 1023) << 10) |
                    ((optix::uint) (N.z * 1023) << 20);

    return packed;
}

static __host__ __device__ __inline__ float packNormalFloat(const optix::float3& normal)
{
    const optix::uint packed = packNormal(normal);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    return __uint_as_float(packed);
#else
    return *((float*) &packed);
#endif
}

static __host__ __device__ __inline__ optix::float3 unpackNormal(optix::uint packed)
{
    optix::float3 N;
    N.x = (float)(packed & 0x000003ff) / 1023;
    N.y = (float)(((packed >> 10) & 0x000003ff)) / 1023;
    N.z = (float)(((packed >> 20) & 0x000003ff)) / 1023;

    return optix::normalize(N * 2.0f - 1.0f);
}

static __host__ __device__ __inline__ optix::float3 unpackNormal(float packed)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    return unpackNormal(__float_as_uint(packed));
#else
    return unpackNormal(*((optix::uint*) &packed));
#endif
}
