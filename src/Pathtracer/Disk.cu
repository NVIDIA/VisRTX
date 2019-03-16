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

#include "Common.h"
#include "Sphere.h"
#include "PackNormals.h"


rtDeclareVariable(int, positionsAndRadii, , );
rtDeclareVariable(int, normals, , ); // packed
rtDeclareVariable(int, vertexcolors, , );
rtDeclareVariable(int, texcoords, , );
rtDeclareVariable(int, animationArray, , );
rtDeclareVariable(float, diskRadius, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(optix::float4, color, attribute color, );
rtDeclareVariable(optix::float3, normal, attribute normal, );
rtDeclareVariable(optix::float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(optix::float2, texCoord, attribute texCoord, );
rtDeclareVariable(int, primIndex, attribute primIndex, );
rtDeclareVariable(float, animationValue, attribute animationValue, );
rtDeclareVariable(MaterialId, material, attribute material, );



RT_FUNCTION void setAttributes(int primitiveIndex, const float3& N)
{
    primIndex = primitiveIndex;
    normal = geometricNormal = N;

    color = optix::make_float4(1.0f);
    texCoord = optix::make_float2(0.0f);
    animationValue = 0.0f;

    material = MATERIAL_NULL;

    if (vertexcolors != RT_BUFFER_ID_NULL)
    {
        const BufferFloat4 colorBuffer(vertexcolors);
        color = colorBuffer[primitiveIndex];
    }

    if (texcoords != RT_BUFFER_ID_NULL)
    {
        const BufferFloat2 texcoordBuffer(texcoords);
        texCoord = texcoordBuffer[primitiveIndex];
    }

    if (animationArray != RT_BUFFER_ID_NULL)
    {
        const BufferFloat animationBuffer(animationArray);
        animationValue = animationBuffer[primitiveIndex];
    }

    // Disks currently do not support per-disk materials
    
}


RT_PROGRAM void DiskIntersect(int prim_idx)
{
    const BufferFloat4 diskBuffer(positionsAndRadii);
    const BufferUint normalBuffer(normals);

    const optix::float4 posRadius = diskBuffer[prim_idx];
    const optix::float3 position = optix::make_float3(posRadius);
    const float radius = posRadius.w * diskRadius;
    const optix::float3 normal = unpackNormal(normalBuffer[prim_idx]);

    // Intersect plane
    float denom = optix::dot(normal, ray.direction);
    if (denom != 0.0f)
    {
        const optix::float3 p = position - ray.origin;
        const float t = optix::dot(p, normal) / denom;

        const optix::float3 hit = ray.origin + t * ray.direction;
        const optix::float3 d = hit - position;

        if (optix::dot(d, d) < radius * radius)
        {
            if (rtPotentialIntersection(t))
            {
                setAttributes(prim_idx, normal);
                rtReportIntersection(0);
            }
        }
    }
}

RT_PROGRAM void DiskBounds(int prim_idx, float result[6])
{
    const BufferFloat4 diskBuffer(positionsAndRadii);

    const optix::float4 posRadius = diskBuffer[prim_idx];

    const optix::float3 center = optix::make_float3(posRadius.x, posRadius.y, posRadius.z);
    const optix::float3 radius = optix::make_float3(posRadius.w * diskRadius);

    optix::Aabb* aabb = (optix::Aabb*)result;

    if (radius.x > 0.0f && !isinf(radius.x))
    {
        aabb->m_min = center - radius;
        aabb->m_max = center + radius;
    }
    else
    {
        aabb->invalidate();
    }
}
