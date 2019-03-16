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


 // Returns uniform random vector along cone with given direction and opening angle.
RT_FUNCTION optix::float3 SampleCone(const optix::float3& dir, float angle, const optix::float2 rand)
{
    // Uniform sampling on north-pole sphere cap
    const float cosAngle = cos(angle);
    const float z = (1.0f - rand.x) * cosAngle + rand.x * 1.0f;
    const float s = sqrtf(1.0f - (z * z));

    const float phi = rand.y * 6.283185f;
    float sins, coss;
    sincos(phi, &sins, &coss);

    optix::float3 vec = optix::make_float3(s * coss, s * sins, z);

    // Transform it to given direction
    optix::Onb onb(dir);
    onb.inverse_transform(vec);

    return vec;
}

RT_FUNCTION optix::float3 SampleDisk(const optix::float3& pos, const optix::float3& dir, float radius, const optix::float2 rand)
{
    const float r = rand.x * radius;
    const float phi = rand.y * 6.28318530718f;

    float sins, coss;
    sincos(phi, &sins, &coss);

    optix::float3 vec = optix::make_float3(r * coss, r * sins, 0.0f);

    // Transform it to given direction and given pos
    optix::Onb onb(dir);
    onb.inverse_transform(vec);

    return pos + vec;
}
