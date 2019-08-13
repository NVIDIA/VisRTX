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
#include "PackNormals.h"

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(optix::float3, hitPoint, attribute hitPoint, );
rtDeclareVariable(optix::float4, color, attribute color, );
rtDeclareVariable(optix::float3, normal, attribute normal, );
rtDeclareVariable(optix::float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(optix::float2, texCoord, attribute texCoord, );
rtDeclareVariable(int, primIndex, attribute primIndex, );
rtDeclareVariable(float, animationValue, attribute animationValue, );
rtDeclareVariable(MaterialId, material, attribute material, );

rtDeclareVariable(int, triangles, , );
rtDeclareVariable(int, vertices, , );
rtDeclareVariable(int, texcoords, , );
rtDeclareVariable(int, vertexcolors, , );
rtDeclareVariable(int, animationArray, , );
rtDeclareVariable(int, materials, , );
rtDeclareVariable(int, perVertexNormals, , );


RT_FUNCTION void setIntersectionAttributes(const int primitiveIndex, const optix::float2 barycentrics)
{
    const float beta = barycentrics.x;
    const float gamma = barycentrics.y;
    const float alpha = 1.0f - beta - gamma;

    const BufferUint4 triangleBuffer(triangles);
    const BufferFloat4 vertexBuffer(vertices);

    const uint4& tri = triangleBuffer[primitiveIndex];

    const float4& v0 = vertexBuffer[tri.x];
    const float4& v1 = vertexBuffer[tri.y];
    const float4& v2 = vertexBuffer[tri.z];

    primIndex = primitiveIndex;

    hitPoint = optix::make_float3(v0) * alpha + optix::make_float3(v1) * beta + optix::make_float3(v2) * gamma;

    geometricNormal = unpackNormal(tri.w);
    if (perVertexNormals)
    {
        const float3 n0 = unpackNormal(v0.w);
        const float3 n1 = unpackNormal(v1.w);
        const float3 n2 = unpackNormal(v2.w);
        normal = optix::normalize(n0 * alpha + n1 * beta + n2 * gamma);
    }
    else
    {
        normal = geometricNormal;
    }

    color = make_float4(1.0f);
    if (vertexcolors != RT_BUFFER_ID_NULL)
    {
        const BufferFloat4 colorBuffer(vertexcolors);
        color = colorBuffer[tri.x] * alpha + colorBuffer[tri.y] * beta + colorBuffer[tri.z] * gamma;
    }

    texCoord = make_float2(0.0);
    if (texcoords != RT_BUFFER_ID_NULL)
    {
        const BufferFloat2 texcoordBuffer(texcoords);
        const float2 t0 = texcoordBuffer[tri.x];
        const float2 t1 = texcoordBuffer[tri.y];
        const float2 t2 = texcoordBuffer[tri.z];
        texCoord = t0 * alpha + t1 * beta + t2 * gamma;
    }

    animationValue = 0.0f;
    if (animationArray != RT_BUFFER_ID_NULL)
    {
        const BufferFloat animationBuffer(animationArray);
        const float v0 = animationBuffer[tri.x];
        const float v1 = animationBuffer[tri.y];
        const float v2 = animationBuffer[tri.z];
        animationValue = v0 * alpha + v1 * beta + v2 * gamma;
    }

    material = MATERIAL_NULL;
    if (materials != RT_BUFFER_ID_NULL)
    {
        const BufferMaterialId materialBuffer(materials);
        material = materialBuffer[primitiveIndex];
    }
}


RT_PROGRAM void TriangleMeshIntersection(int primitiveIndex)
{
    const BufferUint4 triangleBuffer(triangles);
    const BufferFloat4 vertexBuffer(vertices);

    const uint4& tri = triangleBuffer[primitiveIndex];

    const float4& v0 = vertexBuffer[tri.x];
    const float4& v1 = vertexBuffer[tri.y];
    const float4& v2 = vertexBuffer[tri.z];

    const float3 p0 = make_float3(v0);
    const float3 p1 = make_float3(v1);
    const float3 p2 = make_float3(v2);

    float3 n;
    float  t, beta, gamma;
    if (intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma))
    {
        if (rtPotentialIntersection(t))
        {
            setIntersectionAttributes(primitiveIndex, optix::make_float2(beta, gamma));
            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void TriangleMeshBoundingBox(int primitiveIndex, float result[6])
{
    const BufferUint4 triangleBuffer(triangles);
    const BufferFloat4 vertexBuffer(vertices);

    const uint4 tri = triangleBuffer[primitiveIndex];

    const float3 v0 = make_float3(vertexBuffer[tri.x]);
    const float3 v1 = make_float3(vertexBuffer[tri.y]);
    const float3 v2 = make_float3(vertexBuffer[tri.z]);
    const float  area = optix::length(optix::cross(v1 - v0, v2 - v0));

    optix::Aabb* aabb = (optix::Aabb*)result;

    if (area > 0.0f && !isinf(area))
    {
        aabb->m_min = fminf(fminf(v0, v1), v2);
        aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
    }
    else
    {
        aabb->invalidate();
    }
}


#if OPTIX_VERSION_MAJOR >= 6
RT_PROGRAM void TriangleMeshAttribute()
{
    setIntersectionAttributes(rtGetPrimitiveIndex(), rtGetTriangleBarycentrics());
}
#endif
