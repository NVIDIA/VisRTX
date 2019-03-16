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


rtDeclareVariable(int, vertices, , );
rtDeclareVariable(int, lines, , );
rtDeclareVariable(int, texcoords, , );
rtDeclareVariable(int, vertexcolors, , );
rtDeclareVariable(int, animationArray, , );
rtDeclareVariable(int, materials, , );
rtDeclareVariable(float, cylinderRadius, , );

rtDeclareVariable( optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(optix::float4, color, attribute color, );
rtDeclareVariable(optix::float3, normal, attribute normal, );
rtDeclareVariable(optix::float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(optix::float2, texCoord, attribute texCoord, );
rtDeclareVariable(int, primIndex, attribute primIndex, );
rtDeclareVariable(float, animationValue, attribute animationValue, );
rtDeclareVariable(MaterialId, material, attribute material, );


// Explicitly not named Onb to not conflict with the optix::Onb
struct TBN
{
    RT_FUNCTION TBN(const float3& n): normal(n)
    {
        if (fabsf(normal.z) < fabsf(normal.x))
        {
            tangent.x =  normal.z;
            tangent.y =  0.0f;
            tangent.z = -normal.x;
        }
        else
        {
            tangent.x =  0.0f;
            tangent.y =  normal.z;
            tangent.z = -normal.y;
        }
        tangent  = optix::normalize(tangent);
        binormal = optix::cross(normal, tangent);
    }

    RT_FUNCTION float3 transform(const float3& p) const
    {
        return make_float3(optix::dot(p, tangent), optix::dot(p, binormal), optix::dot(p, normal));
    }

    float3 tangent;
    float3 binormal;
    float3 normal;
};

// According to PBRT this is a numerically more stable quadratic solver.
// PBRT explains that the standard quadratic solver is inaccurate for B ~sqrtf(B * B - 4.0f * A * C) due to cancellation errors.
RT_FUNCTION bool quadratic(const float A, const float B, const float C, float& t0, float& t1)
{
    const float d = B * B - 4.0f * A * C;
    if (d < 0.0f)
    {
        return false;
    }
    const float r = sqrtf(d);

    const float q = (B < 0.0f) ? -0.5f * (B - r) : -0.5f * (B + r);

    const float ta = q / A;
    const float tb = C / q;

    if (ta <= tb)
    {
        t0 = ta;
        t1 = tb;
    }
    else
    {
        t0 = tb;
        t1 = ta;
    }
    return true;
}

RT_FUNCTION int intersect_cylinder(const float3& p0, const float3& p1, const float radius, float t[2], float3 normal[2], float ratio[2])
{
    int hits = 0;

    const float3 P = ray.origin;
    const float3 u = ray.direction;

    const float3 Q = p0;
    const float3 axis = p1 - p0;
    const float3 v = optix::normalize(axis);

    if (0.0f < fabsf(optix::dot(u, v))) // No collinear vectors.
    {
        const float3 uxv = optix::cross(u, v);
        const float uxvLength = optix::length(uxv);
        const float distance = fabsf(optix::dot(P - Q, uxv)) / uxvLength;

        if (distance <= radius)
        {
            // Create an orthonormal-basis with Q at the origin and v as z-axis.
            const TBN tbn(v);

            // Transform the ray into the cylinder local coordinate system.
            const float3 origin    = tbn.transform(P - Q);
            const float3 direction = tbn.transform(u);

            const float A = direction.x * direction.x + direction.y * direction.y;
            const float B = 2.0f * (direction.x * origin.x + direction.y * origin.y);
            const float C = origin.x * origin.x + origin.y * origin.y - radius * radius;

            float t0, t1;
            if (quadratic(A, B, C, t0, t1))
            {
                const float height = optix::length(axis);
                const float3 pt0 = origin + t0 * direction;
                const float3 pt1 = origin + t1 * direction;
                if (ray.tmin < t0 && t0 < ray.tmax && 0.0f <= pt0.z && pt0.z <= height)
                {
                    t[hits] = t0;
                    normal[hits] = optix::normalize(pt0.x * tbn.tangent + pt0.y * tbn.binormal);
                    ratio[hits] = pt0.z / height;
                    ++hits;
                }
                if (ray.tmin < t1 && t1 < ray.tmax && 0.0f <= pt1.z && pt1.z <= height)
                {
                    t[hits] = t1;
                    normal[hits] = optix::normalize(pt1.x * tbn.tangent + pt1.y * tbn.binormal);
                    ratio[hits] = pt1.z / height;
                    ++hits;
                }
            }
        }
    }

    return hits;
}


RT_FUNCTION void setAttributes(int primitiveIndex, const float3& N, float ratio)
{
    const BufferInt2 lineBuffer(lines);
    const optix::int2 vertexIdxs = lineBuffer[primitiveIndex];

    primIndex = primitiveIndex;
    normal = geometricNormal = N;

    color = optix::make_float4(1.0f);
    texCoord = optix::make_float2(0.0f);    
    animationValue = 1.0f;    
    material = MATERIAL_NULL;

    if (vertexcolors != RT_BUFFER_ID_NULL)
    {
        const BufferFloat4 colorBuffer(vertexcolors);
        color = (1.0f - ratio) * colorBuffer[vertexIdxs.x] + ratio * colorBuffer[vertexIdxs.y];
    }

    if (texcoords != RT_BUFFER_ID_NULL)
    {
        const BufferFloat2 texcoordBuffer(texcoords);
        texCoord = (1.0f - ratio) * texcoordBuffer[vertexIdxs.x] + ratio * texcoordBuffer[vertexIdxs.y];
    }
        
    if (animationArray != RT_BUFFER_ID_NULL)
    {
        const BufferFloat animationBuffer(animationArray);
        animationValue = (1.0f - ratio) * animationBuffer[vertexIdxs.x] + ratio * animationBuffer[vertexIdxs.y];
    }

    if (materials != RT_BUFFER_ID_NULL)
    {
        const BufferMaterialId materialBuffer(materials);
        material = materialBuffer[primitiveIndex];
    }

}


RT_PROGRAM void CylinderIntersect(int prim_idx)
{
    const BufferInt2 lineBuffer(lines);
    const BufferFloat4 vertexBuffer(vertices);

    int2 vertexIdxs = lineBuffer[prim_idx];
    float4 posAndRadius0 = vertexBuffer[vertexIdxs.x];
    float4 posAndRadius1 = vertexBuffer[vertexIdxs.y];

    float3 pos0 = make_float3(posAndRadius0.x, posAndRadius0.y, posAndRadius0.z);
    float3 pos1 = make_float3(posAndRadius1.x, posAndRadius1.y, posAndRadius1.z);

    const float radius = (posAndRadius0.w + posAndRadius1.w) * 0.5f * cylinderRadius;

    float t[2];
    float3 n[2];
    float ratio[2];
    int intersections = intersect_cylinder(pos0, pos1, radius, t, n, ratio);

    if (intersections >= 1)
    {
        if (rtPotentialIntersection(t[0]))
        {
            setAttributes(prim_idx, n[0], ratio[0]);
            rtReportIntersection(0);
        }

        if (intersections == 2)
        {
            if (rtPotentialIntersection(t[1]))
            {
                setAttributes(prim_idx, n[1], ratio[1]);
                rtReportIntersection(0);
            }
        }
    }
}


RT_PROGRAM void CylinderBounds (int prim_idx, float result[6])
{
    const BufferInt2 lineBuffer(lines);
    const BufferFloat4 vertexBuffer(vertices);

    int2 vertexIdxs = lineBuffer[prim_idx];
    float4 posAndRadius0 = vertexBuffer[vertexIdxs.x];
    float4 posAndRadius1 = vertexBuffer[vertexIdxs.y];

    float3 pos0 = make_float3(posAndRadius0.x, posAndRadius0.y, posAndRadius0.z);
    float3 pos1 = make_float3(posAndRadius1.x, posAndRadius1.y, posAndRadius1.z);

    const float radius = (posAndRadius0.w + posAndRadius1.w) * 0.5f * cylinderRadius;

    optix::Aabb* aabb = (optix::Aabb*) result;

    if(radius > 0.0f && !isinf(radius))
    {
        aabb->m_min = fminf(pos0, pos1) - radius;
        aabb->m_max = fmaxf(pos0, pos1) + radius;
    }
    else
    {
        aabb->invalidate();
    }
}
