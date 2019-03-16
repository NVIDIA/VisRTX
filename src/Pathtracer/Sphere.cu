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


rtDeclareVariable(int, spheres, , );
rtDeclareVariable(int, texcoords, , );
rtDeclareVariable(int, vertexcolors, , );
rtDeclareVariable(int, animationArray, , );
rtDeclareVariable(int, materials, , );

rtDeclareVariable(float, sphereRadius, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(optix::float4, color, attribute color, );
rtDeclareVariable(optix::float3, normal, attribute normal, );
rtDeclareVariable(optix::float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(optix::float2, texCoord, attribute texCoord, );
rtDeclareVariable(int, primIndex, attribute primIndex, );
rtDeclareVariable(float, animationValue, attribute animationValue, );
rtDeclareVariable(MaterialId, material, attribute material, );


RT_FUNCTION void setAttributes(int primitiveIndex, const optix::float3& N)
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

    if (materials != RT_BUFFER_ID_NULL)
    {
        const BufferMaterialId materialBuffer(materials);
        material = materialBuffer[primitiveIndex];
    }
}


RT_PROGRAM void SphereIntersect(int prim_idx)
{
    const BufferFloat4 sphereBuffer(spheres);

    optix::float4 sphereAndRadius = sphereBuffer[prim_idx];
    const optix::float3 center = make_float3(sphereAndRadius.x, sphereAndRadius.y, sphereAndRadius.z);
    const float radius = sphereAndRadius.w * sphereRadius;

    float t0, t1;
    if (intersectSphere(center, radius, ray.origin, ray.direction, t0, t1))
    {
        // First hit        
        if (rtPotentialIntersection(t0))
        {
            const optix::float3 n0 = ((ray.origin + t0 * ray.direction) - center) / radius;
            setAttributes(prim_idx, n0);

            if (rtReportIntersection(0))
                return;
        }

        // Second hit        
        if (rtPotentialIntersection(t1))
        {
            const optix::float3 n1 = ((ray.origin + t1 * ray.direction) - center) / radius;
            setAttributes(prim_idx, n1);

            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void SphereBounds(int prim_idx, float result[6])
{
    const BufferFloat4 sphereBuffer(spheres);
    const optix::float4 sphereAndRadius = sphereBuffer[prim_idx];

    const optix::float3 center = optix::make_float3(sphereAndRadius.x, sphereAndRadius.y, sphereAndRadius.z);
    const optix::float3 radius = optix::make_float3(sphereAndRadius.w * sphereRadius);

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
