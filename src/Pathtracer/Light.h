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



 /*
  * Common light struct used by all light types (no callable program per light type for now; more inlining!).
  */
class Light
{
public:
    enum
    {
        POSITIONAL = 0,
        DIRECTIONAL = 1,
        QUAD = 2,
        SPOT = 3,
        AMBIENT = 4,
        HDRI = 5,
    };

    optix::float3 color;
    optix::float3 pos;
    optix::float3 dir;
    optix::float3 up;
    optix::float3 edge1;
    optix::float3 edge2;

    int id;
    int type;
    int visible;
    float radius;
    float angularDiameter; // half opening angle actually (angle between cone axis and surface)        
    float outerAngle;
    float innerAngle;
    int twoSided;        
    int texture;
    float pdf; // constant pre-computable part of pdf

    // Lights are stored in arrays, so pad to 16 byte struct alignment
    // 6 * 12 (float3) + 10 * 4 (int/float) = 112
    // -> 0 bytes padding (112 % 16 = 0)
    //char padding[0];
};
