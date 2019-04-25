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

#include "Geometry.h"
#include "Material.h"
#include "ProgramLoader.h"

#include "Pathtracer/PackNormals.h"
#include "Pathtracer/Common.h"

#include <cstring>
#include <cassert>

namespace VisRTX
{
    namespace Impl
    {
        // ------------------------------------------------------------------------------------
        // Base class
        // ------------------------------------------------------------------------------------
        Geometry::Geometry(bool isTriangles)
        {
            this->isTriangles = isTriangles;

            optix::Context context = OptiXContext::Get();
            ProgramLoader& loader = ProgramLoader::Get();
            this->defaultMaterial = context->createMaterial();
            this->defaultMaterial->setClosestHitProgram(RADIANCE_RAY_TYPE, loader.closestHitProgram);
            this->defaultMaterial->setAnyHitProgram(RADIANCE_RAY_TYPE, loader.anyHitProgram);
            this->defaultMaterial->setAnyHitProgram(OCCLUSION_RAY_TYPE, loader.anyHitOcclusionProgram);
            this->defaultMaterial->setClosestHitProgram(PICK_RAY_TYPE, loader.closestHitPickProgram);
            this->defaultMaterial->setAnyHitProgram(PICK_RAY_TYPE, loader.anyHitProgram);

            this->instance = context->createGeometryInstance();
            this->SetMaterial(this->defaultMaterial, false);

            // Assign next free id
            static int geometryId = 0;
            this->id = ++geometryId;
            this->instance["id"]->setInt(this->id);

            this->SetAnimatedParameterization(false, 0.0f, 0.0f, 0.0f);
        }

        Geometry::~Geometry()
        {
            if (this->materialHandle)
                this->materialHandle->Release();

            this->UpdatePrimitiveMaterialHandles(0, nullptr);

            // Destroy OptiX objects
            Destroy(this->instance.get());
            Destroy(this->defaultMaterial.get());
        }

        void Geometry::SetMaterial(VisRTX::Material* material)
        {
            if (material == this->materialHandle)
                return;

            // Free current material
            if (this->materialHandle)
                this->materialHandle->Release();

            // Assign ID of callable program
            VisRTX::Impl::Material* mat = dynamic_cast<VisRTX::Impl::Material*>(material);
            assert(mat != nullptr);

            this->materialHandle = material; // Store handle and increase ref count
            this->materialHandle->Retain();

            this->instance["geometryMaterial"]->setUserData(sizeof(MaterialId), &mat->material);
        }

        void Geometry::SetAnimatedParameterization(bool enabled, float time, float frequency, float scale)
        {
            this->instance["animateSurface"]->setInt(enabled ? 1 : 0);
            this->instance["animationTime"]->setFloat(time);
            this->instance["animationFrequency"]->setFloat(frequency);
            this->instance["animationScale"]->setFloat(scale);
        }

        void Geometry::AddAcceleration(optix::Acceleration acceleration)
        {
            this->accelerations.insert(acceleration);
        }

        void Geometry::RemoveAcceleration(optix::Acceleration acceleration)
        {
            this->accelerations.erase(acceleration);
        }

        void Geometry::MarkDirty()
        {
#if OPTIX_VERSION_MAJOR >= 6
            if (!this->isTriangles)
                this->instance->getGeometry()->markDirty();
#else
            this->instance->getGeometry()->markDirty();
#endif

            for (optix::Acceleration acc : this->accelerations)
                acc->markDirty();
        }

        void Geometry::SetMaterial(optix::Material material, bool isLight)
        {
            this->instance->setMaterialCount(1);
            this->instance->setMaterial(0, material);
        }

        void Geometry::UpdatePrimitiveMaterialHandles(uint32_t numMaterials, VisRTX::Material** const materials)
        {
            // Release current materials
            for (VisRTX::Material* mat : this->primitiveMaterialHandles)
                if (mat)
                    mat->Release();

            this->primitiveMaterialHandles.resize(numMaterials);

            // Store new handles and increase ref count
            for (uint32_t i = 0; i < numMaterials; ++i)
            {
                VisRTX::Material* mat = materials ? materials[i] : nullptr;

                this->primitiveMaterialHandles[i] = mat;

                if (mat)
                    mat->Retain();
            }
        }


        // ------------------------------------------------------------------------------------
        // Triangle geometry
        // ------------------------------------------------------------------------------------
        TriangleGeometry::TriangleGeometry() : Geometry(true)
        {
            optix::Context context = OptiXContext::Get();

#if OPTIX_VERSION_MAJOR >= 6
            this->geometry = context->createGeometryTriangles();
            this->instance->setGeometryTriangles(this->geometry);

            ProgramLoader& loader = ProgramLoader::Get();
            this->geometry->setAttributeProgram(loader.triangleAttributeProgram);
#else
            this->geometry = context->createGeometry();
            this->instance->setGeometry(this->geometry);

            ProgramLoader& loader = ProgramLoader::Get();
            this->geometry->setIntersectionProgram(loader.triangleIsectProgram);
            this->geometry->setBoundingBoxProgram(loader.triangleBoundsProgram);
#endif

            // Assign empty dummy buffers
            this->SetColors(nullptr);
            this->SetTexCoords(nullptr);
            this->SetParameterization(nullptr);
            this->SetMaterials(nullptr);
        }

        TriangleGeometry::TriangleGeometry(uint32_t numTriangles, const Vec3ui* triangles, uint32_t numVertices, const Vec3f* vertices, const Vec3f* normals) : TriangleGeometry()
        {
            this->SetTriangles(numTriangles, triangles, numVertices, vertices, normals);
        }

        TriangleGeometry::~TriangleGeometry()
        {
            Destroy(this->vertexBuffer.get());
            Destroy(this->triangleBuffer.get());
            Destroy(this->colorBuffer.get());
            Destroy(this->texCoordBuffer.get());
            Destroy(this->animationValuesBuffer.get());
            Destroy(this->materialBuffer.get());
        }

        void TriangleGeometry::SetTriangles(uint32_t numTriangles, const Vec3ui* triangles, uint32_t numVertices, const Vec3f* vertices, const Vec3f* normals)
        {
            if (this->numVertices > 0 && numVertices != this->numVertices)
            {
                // Reset vertex attribute arrays if the vertex count has changed
                this->SetColors(nullptr);
                this->SetTexCoords(nullptr);
                this->SetParameterization(nullptr);
                this->SetMaterials(nullptr);
            }

            this->numVertices = numVertices;
            this->numTriangles = numTriangles;

            optix::Context context = OptiXContext::Get();

            // Vertices + vertex normals
            if (!this->vertexBuffer)
                this->vertexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, numVertices);
            else
                this->vertexBuffer->setSize(numVertices);

            optix::float4* vertexData = reinterpret_cast<optix::float4*>(vertexBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

#pragma omp parallel for
            for (int64_t i = 0; i < numVertices; ++i)
            {
                const optix::float3 pos = make_float3(vertices[i]);
                float packedNormal = normals ? packNormalFloat(make_float3(normals[i])) : 0.0f;
                vertexData[i] = optix::make_float4(pos, packedNormal);
            }

            // Triangles + geometric normals
            if (!this->triangleBuffer)
                this->triangleBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT4, numTriangles);
            else
                this->triangleBuffer->setSize(numTriangles);

            optix::uint4* triangleData = reinterpret_cast<optix::uint4*>(triangleBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

#pragma omp parallel for
            for (int64_t i = 0; i < numTriangles; i++)
            {
                const optix::uint3& triangle = make_uint3(triangles[i]);

                // Compute geometric normal
                const optix::float3& p0 = optix::make_float3(vertexData[triangle.x]);
                const optix::float3& p1 = optix::make_float3(vertexData[triangle.y]);
                const optix::float3& p2 = optix::make_float3(vertexData[triangle.z]);

                const optix::float3 Ng = optix::normalize(optix::cross(p1 - p0, p2 - p0));

                // Store indices and packed normal
                uint packedNormal = packNormal(Ng);
                triangleData[i] = optix::make_uint4(triangle, packedNormal);
            }

            vertexBuffer->unmap();
            triangleBuffer->unmap();

            this->geometry["perVertexNormals"]->setInt(normals != nullptr);

            this->geometry["vertices"]->setInt(vertexBuffer->getId());
            this->geometry["triangles"]->setInt(triangleBuffer->getId());

#if OPTIX_VERSION_MAJOR >= 6
            this->geometry->setVertices(this->numVertices, vertexBuffer, 0, sizeof(optix::float4), RT_FORMAT_FLOAT3);
            this->geometry->setTriangleIndices(triangleBuffer, 0, sizeof(optix::uint4), RT_FORMAT_UNSIGNED_INT3);
#endif

            this->geometry->setPrimitiveCount(this->numTriangles);

            this->MarkDirty();
        }

        void TriangleGeometry::SetColors(const Vec4f* vertexColors)
        {
            if (vertexColors)
            {
                optix::Context context = OptiXContext::Get();

                if (!colorBuffer)
                    this->colorBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, vertexColors ? this->numVertices : 0);
                else
                    this->colorBuffer->setSize(vertexColors ? this->numVertices : 0);

                memcpy(colorBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertexColors, this->numVertices * sizeof(optix::float4));
                colorBuffer->unmap();

                this->geometry["vertexcolors"]->setInt(colorBuffer->getId());
            }
            else
            {
                this->geometry["vertexcolors"]->setInt(RT_BUFFER_ID_NULL);
            }
        }

        void TriangleGeometry::SetTexCoords(const Vec2f* vertexTexCoords)
        {
            if (vertexTexCoords)
            {
                optix::Context context = OptiXContext::Get();

                if (!this->texCoordBuffer)
                    this->texCoordBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, vertexTexCoords ? this->numVertices : 0);
                else
                    this->texCoordBuffer->setSize(vertexTexCoords ? this->numVertices : 0);

                memcpy(texCoordBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertexTexCoords, this->numVertices * sizeof(optix::float2));
                texCoordBuffer->unmap();

                this->geometry["texcoords"]->setInt(texCoordBuffer->getId());
            }
            else
            {
                this->geometry["texcoords"]->setInt(RT_BUFFER_ID_NULL);
            }
        }

        void TriangleGeometry::SetParameterization(const float* vertexParameterization)
        {
            if (vertexParameterization)
            {
                optix::Context context = OptiXContext::Get();

                if (!animationValuesBuffer)
                    this->animationValuesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, vertexParameterization ? this->numVertices : 0);
                else
                    this->animationValuesBuffer->setSize(vertexParameterization ? this->numVertices : 0);

                memcpy(animationValuesBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertexParameterization, this->numVertices * sizeof(float));
                animationValuesBuffer->unmap();

                this->geometry["animationArray"]->setInt(animationValuesBuffer->getId());
            }
            else
            {
                this->geometry["animationArray"]->setInt(RT_BUFFER_ID_NULL);
            }
        }

        void TriangleGeometry::SetMaterials(VisRTX::Material** const triangleMaterials)
        {
            if (triangleMaterials)
            {
                optix::Context context = OptiXContext::Get();

                if (!this->materialBuffer)
                    this->materialBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, triangleMaterials ? this->numTriangles : 0);
                else
                    this->materialBuffer->setSize(triangleMaterials ? this->numTriangles : 0);

                MaterialId* dst = reinterpret_cast<MaterialId*>(this->materialBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
                for (uint32_t i = 0; i < this->numTriangles; ++i)
                {
                    const VisRTX::Impl::Material* mat = dynamic_cast<const VisRTX::Impl::Material*>(triangleMaterials[i]);
                    assert(mat != nullptr);
                    dst[i] = mat->material;
                }
                this->materialBuffer->unmap();

                this->geometry["materials"]->setInt(materialBuffer->getId());
            }
            else
            {
                this->geometry["materials"]->setInt(RT_BUFFER_ID_NULL);
            }

            this->UpdatePrimitiveMaterialHandles(this->numTriangles, triangleMaterials);
        }


        // ------------------------------------------------------------------------------------
        // Sphere geometry
        // ------------------------------------------------------------------------------------
        SphereGeometry::SphereGeometry() : Geometry(false)
        {
            optix::Context context = OptiXContext::Get();
            this->geometry = context->createGeometry();
            this->instance->setGeometry(this->geometry);

            ProgramLoader& loader = ProgramLoader::Get();
            this->geometry->setIntersectionProgram(loader.sphereIsectProgram);
            this->geometry->setBoundingBoxProgram(loader.sphereBoundsProgram);

            // Assign empty dummy buffers
            this->SetColors(nullptr);
            this->SetTexCoords(nullptr);
            this->SetParameterization(nullptr);
            this->SetMaterials(nullptr);

            this->SetRadius(1.0f);
        }

        SphereGeometry::SphereGeometry(uint32_t numVertices, const Vec3f* vertices, const float* radii) : SphereGeometry()
        {
            this->SetSpheres(numVertices, vertices, radii);
        }

        SphereGeometry::~SphereGeometry()
        {
            Destroy(this->sphereBuffer.get());
            Destroy(this->colorBuffer.get());
            Destroy(this->texCoordBuffer.get());
            Destroy(this->animationValuesBuffer.get());
            Destroy(this->materialBuffer.get());
        }

        void SphereGeometry::SetSpheres(uint32_t numVertices, const Vec3f* vertices, const float* radii)
        {
            if (this->numVertices > 0 && numVertices != this->numVertices)
            {
                // Reset vertex attribute arrays if the vertex count has changed
                this->SetColors(nullptr);
                this->SetTexCoords(nullptr);
                this->SetParameterization(nullptr);
                this->SetMaterials(nullptr);
            }

            this->numVertices = numVertices;

            if (vertices != nullptr)
            {
                optix::Context context = OptiXContext::Get();

                // Spheres buffer
                if (!this->sphereBuffer)
                    this->sphereBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, numVertices);
                else
                    this->sphereBuffer->setSize(numVertices);

                optix::float4* sphereData = reinterpret_cast<optix::float4*>(this->sphereBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

                for (uint32_t i = 0; i < numVertices; ++i)
                {
                    const optix::float3 pos = make_float3(vertices[i]);
                    float radius = (radii != nullptr) ? radii[i] : 1.0f;
                    sphereData[i] = optix::make_float4(pos, radius);
                }

                this->sphereBuffer->unmap();

                this->instance["spheres"]->setInt(sphereBuffer->getId());
            }
            else
            {
                this->numVertices = 0;

                this->instance["spheres"]->setInt(RT_BUFFER_ID_NULL);
            }

            this->geometry->setPrimitiveCount(this->numVertices);

            this->MarkDirty();
        }

        void SphereGeometry::SetColors(const Vec4f* vertexColors)
        {
            if (vertexColors)
            {
                optix::Context context = OptiXContext::Get();

                if (!colorBuffer)
                    colorBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, vertexColors ? this->numVertices : 0);
                else
                    this->colorBuffer->setSize(vertexColors ? this->numVertices : 0);

                memcpy(colorBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertexColors, this->numVertices * sizeof(optix::float4));
                colorBuffer->unmap();

                this->instance["vertexcolors"]->setInt(colorBuffer->getId());
            }
            else
            {
                this->instance["vertexcolors"]->setInt(RT_BUFFER_ID_NULL);
            }
        }

        void SphereGeometry::SetTexCoords(const Vec2f* vertexTexCoords)
        {
            if (vertexTexCoords)
            {
                optix::Context context = OptiXContext::Get();

                if (!texCoordBuffer)
                    texCoordBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, vertexTexCoords ? this->numVertices : 0);
                else
                    this->texCoordBuffer->setSize(vertexTexCoords ? this->numVertices : 0);

                memcpy(texCoordBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertexTexCoords, this->numVertices * sizeof(optix::float2));
                texCoordBuffer->unmap();

                this->instance["texcoords"]->setInt(texCoordBuffer->getId());
            }
            else
            {
                this->instance["texcoords"]->setInt(RT_BUFFER_ID_NULL);
            }
        }

        void SphereGeometry::SetParameterization(const float* vertexParameterization)
        {
            if (vertexParameterization)
            {
                optix::Context context = OptiXContext::Get();

                if (!this->animationValuesBuffer)
                    this->animationValuesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, vertexParameterization ? this->numVertices : 0);
                else
                    this->animationValuesBuffer->setSize(vertexParameterization ? this->numVertices : 0);

                memcpy(animationValuesBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertexParameterization, this->numVertices * sizeof(float));
                animationValuesBuffer->unmap();

                this->instance["animationArray"]->setInt(animationValuesBuffer->getId());
            }
            else
            {
                this->instance["animationArray"]->setInt(RT_BUFFER_ID_NULL);
            }
        }

        void SphereGeometry::SetRadius(float radius)
        {
            this->instance["sphereRadius"]->setFloat(radius);
            this->MarkDirty();
        }

        void SphereGeometry::SetSpheresAndColors(uint32_t numVertices, const void* vertexData, uint32_t bytesPerSphere, uint32_t offsetCenter, int32_t offsetRadius, int32_t offsetColorIndex, const Vec4f* colors)
        {
            if (this->numVertices > 0 && numVertices != this->numVertices)
            {
                // Reset vertex attribute arrays if the vertex count has changed
                this->SetColors(nullptr);
                this->SetTexCoords(nullptr);
                this->SetParameterization(nullptr);
                this->SetMaterials(nullptr);
            }

            this->numVertices = numVertices;

            if (vertexData)
            {
                optix::Context context = OptiXContext::Get();

                // Spheres buffer
                if (!this->sphereBuffer)
                    this->sphereBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, numVertices);
                else
                    this->sphereBuffer->setSize(numVertices);

                // Color buffer
                if (!this->colorBuffer)
                    this->colorBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, colors ? this->numVertices : 0);
                else
                    this->colorBuffer->setSize(colors ? this->numVertices : 0);

                optix::float4* sphereData = reinterpret_cast<optix::float4*>(this->sphereBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
                optix::float4* colorData = colors ? reinterpret_cast<optix::float4*>(this->colorBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD)) : nullptr;

#pragma omp parallel for
                for (int32_t i = 0; i < static_cast<int32_t>(numVertices); ++i)
                {
                    const uint8_t* base = reinterpret_cast<const uint8_t*>(vertexData) + i * bytesPerSphere;

                    const optix::float3 pos = make_float3(*reinterpret_cast<const VisRTX::Vec3f*>(base + offsetCenter));
                    float radius = (offsetRadius < 0) ? 1.0f : *reinterpret_cast<const float*>(base + offsetRadius);
                    sphereData[i] = optix::make_float4(pos, radius);

                    if (colors)
                    {
                        uint32_t colorIndex = (offsetColorIndex < 0) ? i : *reinterpret_cast<const int32_t*>(base + offsetColorIndex);
                        colorData[i] = make_float4(colors[colorIndex]);
                    }
                }

                this->sphereBuffer->unmap();
                if (colors != nullptr)
                    this->colorBuffer->unmap();

                this->instance["spheres"]->setInt(sphereBuffer->getId());
                this->instance["vertexcolors"]->setInt((colors != nullptr) ? this->colorBuffer->getId() : RT_BUFFER_ID_NULL);
            }
            else
            {
                this->numVertices = 0;

                this->instance["spheres"]->setInt(RT_BUFFER_ID_NULL);
                this->instance["vertexcolors"]->setInt(RT_BUFFER_ID_NULL);
            }

            this->geometry->setPrimitiveCount(this->numVertices);

            this->MarkDirty();
        }

        void SphereGeometry::SetMaterials(VisRTX::Material** const sphereMaterials)
        {
            if (sphereMaterials)
            {
                optix::Context context = OptiXContext::Get();

                if (!this->materialBuffer)
                    this->materialBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, sphereMaterials ? this->numVertices : 0);
                else
                    this->materialBuffer->setSize(sphereMaterials ? this->numVertices : 0);

                MaterialId* dst = reinterpret_cast<MaterialId*>(this->materialBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
                for (uint32_t i = 0; i < this->numVertices; ++i)
                {
                    const VisRTX::Impl::Material* mat = dynamic_cast<const VisRTX::Impl::Material*>(sphereMaterials[i]);
                    assert(mat != nullptr);
                    dst[i] = mat->material;
                }
                this->materialBuffer->unmap();

                this->instance["materials"]->setInt(materialBuffer->getId());
            }
            else
            {
                this->instance["materials"]->setInt(RT_BUFFER_ID_NULL);
            }

            this->UpdatePrimitiveMaterialHandles(this->numVertices, sphereMaterials);
        }


        // ------------------------------------------------------------------------------------
        // Cylinder geometry
        // ------------------------------------------------------------------------------------
        CylinderGeometry::CylinderGeometry() : Geometry(false)
        {
            optix::Context context = OptiXContext::Get();
            this->geometry = context->createGeometry();
            this->instance->setGeometry(this->geometry);

            ProgramLoader& loader = ProgramLoader::Get();
            this->geometry->setIntersectionProgram(loader.cylinderIsectProgram);
            this->geometry->setBoundingBoxProgram(loader.cylinderBoundsProgram);

            // Assign empty dummy buffers
            this->SetColors(nullptr);
            this->SetTexCoords(nullptr);
            this->SetParameterization(nullptr);
            this->SetMaterials(nullptr);

            this->SetRadius(1.0f);
        }

        CylinderGeometry::CylinderGeometry(uint32_t numCylinders, const Vec2ui* cylinders, uint32_t numVertices, const Vec3f* vertices, const float* radii) : CylinderGeometry()
        {
            this->SetCylinders(numCylinders, cylinders, numVertices, vertices, radii);
        }

        CylinderGeometry::~CylinderGeometry()
        {
            Destroy(this->vertexBuffer.get());
            Destroy(this->lineBuffer.get());
            Destroy(this->colorBuffer.get());
            Destroy(this->texCoordBuffer.get());
            Destroy(this->animationValuesBuffer.get());
            Destroy(this->materialBuffer.get());
        }

        void CylinderGeometry::SetCylinders(uint32_t numCylinders, const Vec2ui* cylinders, uint32_t numVertices, const Vec3f* vertices, const float* radii)
        {
            if (this->numVertices > 0 && numVertices != this->numVertices)
            {
                // Reset vertex attribute arrays if the vertex count has changed
                this->SetColors(nullptr);
                this->SetTexCoords(nullptr);
                this->SetParameterization(nullptr);
                this->SetMaterials(nullptr);
            }

            this->numVertices = numVertices;
            this->numCylinders = numCylinders;

            if (cylinders != nullptr && vertices != nullptr)
            {
                optix::Context context = OptiXContext::Get();

                // Vertex buffer
                if (!this->vertexBuffer)
                    this->vertexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, numVertices);
                else
                    this->vertexBuffer->setSize(numVertices);

                optix::float4* vertexData = reinterpret_cast<optix::float4*>(this->vertexBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

                for (uint32_t i = 0; i < numVertices; ++i)
                {
                    const optix::float3 pos = make_float3(vertices[i]);
                    float radius = (radii != nullptr) ? radii[i] : 1.0f;
                    vertexData[i] = optix::make_float4(pos, radius);
                }

                this->vertexBuffer->unmap();

                // Lines buffer
                if (!this->lineBuffer)
                    this->lineBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT2, numCylinders);
                else
                    this->lineBuffer->setSize(numCylinders);

                memcpy(this->lineBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), cylinders, numCylinders * sizeof(optix::uint2));
                this->lineBuffer->unmap();

                this->instance["vertices"]->setInt(this->vertexBuffer->getId());
                this->instance["lines"]->setInt(this->lineBuffer->getId());
            }
            else
            {
                this->numVertices = 0;
                this->numCylinders = 0;

                this->instance["vertices"]->setInt(RT_BUFFER_ID_NULL);
                this->instance["lines"]->setInt(RT_BUFFER_ID_NULL);
            }

            this->geometry->setPrimitiveCount(this->numCylinders);

            this->MarkDirty();
        }

        void CylinderGeometry::SetColors(const Vec4f* vertexColors)
        {
            optix::Context context = OptiXContext::Get();

            if (vertexColors)
            {
                if (!this->colorBuffer)
                    this->colorBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, vertexColors ? this->numVertices : 0);
                else
                    this->colorBuffer->setSize(vertexColors ? this->numVertices : 0);

                memcpy(colorBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertexColors, this->numVertices * sizeof(optix::float4));
                colorBuffer->unmap();

                this->instance["vertexcolors"]->setInt(this->colorBuffer->getId());
            }
            else
            {
                this->instance["vertexcolors"]->setInt(RT_BUFFER_ID_NULL);
            }
        }

        void CylinderGeometry::SetTexCoords(const Vec2f* vertexTexCoords)
        {
            if (vertexTexCoords)
            {
                optix::Context context = OptiXContext::Get();

                if (!this->texCoordBuffer)
                    this->texCoordBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, vertexTexCoords ? this->numVertices : 0);
                else
                    this->texCoordBuffer->setSize(vertexTexCoords ? this->numVertices : 0);

                memcpy(texCoordBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertexTexCoords, this->numVertices * sizeof(optix::float2));
                texCoordBuffer->unmap();

                this->instance["texcoords"]->setInt(texCoordBuffer->getId());
            }
            else
            {
                this->instance["texcoords"]->setInt(RT_BUFFER_ID_NULL);
            }
        }

        void CylinderGeometry::SetParameterization(const float* vertexParameterization)
        {
            if (vertexParameterization)
            {
                optix::Context context = OptiXContext::Get();

                if (!this->animationValuesBuffer)
                    this->animationValuesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, vertexParameterization ? this->numVertices : 0);
                else
                    this->animationValuesBuffer->setSize(vertexParameterization ? this->numVertices : 0);

                memcpy(animationValuesBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertexParameterization, this->numVertices * sizeof(float));
                animationValuesBuffer->unmap();

                this->instance["animationArray"]->setInt(animationValuesBuffer->getId());
            }
            else
            {
                this->instance["animationArray"]->setInt(RT_BUFFER_ID_NULL);
            }
        }

        void CylinderGeometry::SetRadius(float radius)
        {
            this->instance["cylinderRadius"]->setFloat(radius);
            this->MarkDirty();
        }

        void CylinderGeometry::SetCylindersAndColors(uint32_t numCylinders, const void* cylinderData, uint32_t bytesPerCylinder, uint32_t offset_v0, uint32_t offset_v1, int32_t offsetRadius, const Vec4f* cylinderColors)
        {
            if (this->numVertices > 0 && (numCylinders * 2) != this->numVertices)
            {
                // Reset vertex attribute arrays if the vertex count has changed
                this->SetColors(nullptr);
                this->SetTexCoords(nullptr);
                this->SetParameterization(nullptr);
                this->SetMaterials(nullptr);
            }

            this->numVertices = (numCylinders * 2);
            this->numCylinders = numCylinders;

            if (cylinderData != nullptr)
            {
                optix::Context context = OptiXContext::Get();

                // Vertex buffer
                if (!this->vertexBuffer)
                    this->vertexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, this->numVertices);
                else
                    this->vertexBuffer->setSize(this->numVertices);

                // Lines buffer
                if (!lineBuffer)
                    this->lineBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT2, numCylinders);
                else
                    this->lineBuffer->setSize(numCylinders);

                // Color buffer
                if (!this->colorBuffer)
                    this->colorBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, cylinderColors ? this->numVertices : 0);
                else
                    this->colorBuffer->setSize(cylinderColors ? this->numVertices : 0);

                optix::float4* vertexData = reinterpret_cast<optix::float4*>(vertexBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
                optix::uint2* lineData = reinterpret_cast<optix::uint2*>(lineBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
                optix::float4* colorData = cylinderColors ? reinterpret_cast<optix::float4*>(this->colorBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD)) : nullptr;

#pragma omp parallel for
                for (int32_t i = 0; i < static_cast<int32_t>(numCylinders); ++i)
                {
                    const uint8_t* base = reinterpret_cast<const uint8_t*>(cylinderData) + i * bytesPerCylinder;

                    const optix::float3 v0 = make_float3(*reinterpret_cast<const VisRTX::Vec3f*>(base + offset_v0));
                    const optix::float3 v1 = make_float3(*reinterpret_cast<const VisRTX::Vec3f*>(base + offset_v1));
                    float radius = (offsetRadius < 0) ? 1.0f : *reinterpret_cast<const float*>(base + offsetRadius);

                    vertexData[2 * i] = optix::make_float4(v0, radius);
                    vertexData[2 * i + 1] = optix::make_float4(v1, radius);

                    lineData[i] = optix::make_uint2(2 * i, 2 * i + 1);

                    if (cylinderColors)
                    {
                        const Vec4f& color = cylinderColors[i];
                        colorData[2 * i] = make_float4(color);
                        colorData[2 * i + 1] = make_float4(color);
                    }
                }

                vertexBuffer->unmap();
                lineBuffer->unmap();
                if (cylinderColors)
                    this->colorBuffer->unmap();

                this->instance["vertices"]->setInt(this->vertexBuffer->getId());
                this->instance["lines"]->setInt(this->lineBuffer->getId());

                if (cylinderColors)
                    this->instance["vertexcolors"]->setInt(this->colorBuffer->getId());
                else
                    this->instance["vertexcolors"]->setInt(RT_BUFFER_ID_NULL);
            }
            else
            {
                this->instance["vertices"]->setInt(RT_BUFFER_ID_NULL);
                this->instance["lines"]->setInt(RT_BUFFER_ID_NULL);
                this->instance["vertexcolors"]->setInt(RT_BUFFER_ID_NULL);
            }

            this->geometry->setPrimitiveCount(numCylinders);

            this->MarkDirty();
        }

        void CylinderGeometry::SetMaterials(VisRTX::Material** const cylinderMaterials)
        {
            if (cylinderMaterials)
            {
                optix::Context context = OptiXContext::Get();

                if (!this->materialBuffer)
                    this->materialBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, cylinderMaterials ? this->numCylinders : 0);
                else
                    this->materialBuffer->setSize(cylinderMaterials ? this->numCylinders : 0);

                MaterialId* dst = reinterpret_cast<MaterialId*>(this->materialBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
                for (uint32_t i = 0; i < this->numCylinders; ++i)
                {
                    const VisRTX::Impl::Material* mat = dynamic_cast<const VisRTX::Impl::Material*>(cylinderMaterials[i]);
                    assert(mat != nullptr);
                    dst[i] = mat->material;
                }
                this->materialBuffer->unmap();

                this->instance["materials"]->setInt(materialBuffer->getId());
            }
            else
            {
                this->instance["materials"]->setInt(RT_BUFFER_ID_NULL);
            }

            this->UpdatePrimitiveMaterialHandles(this->numCylinders, cylinderMaterials);
        }

        // ------------------------------------------------------------------------------------
        // Disk geometry
        // ------------------------------------------------------------------------------------
        DiskGeometry::DiskGeometry() : Geometry(false)
        {
            optix::Context context = OptiXContext::Get();
            this->geometry = context->createGeometry();
            this->instance->setGeometry(this->geometry);

            ProgramLoader& loader = ProgramLoader::Get();
            this->geometry->setIntersectionProgram(loader.diskIsectProgram);
            this->geometry->setBoundingBoxProgram(loader.diskBoundsProgram);

            // Assign empty dummy buffers
            this->SetColors(nullptr);
            this->SetTexCoords(nullptr);
            this->SetParameterization(nullptr);

            this->SetRadius(1.0f);
        }

        DiskGeometry::DiskGeometry(uint32_t numVertices, const Vec3f* vertices, const Vec3f* normals, const float* radii) : DiskGeometry()
        {
            this->SetDisks(numVertices, vertices, normals, radii);
        }

        DiskGeometry::~DiskGeometry()
        {
            Destroy(this->diskBuffer.get());
            Destroy(this->normalBuffer.get());
            Destroy(this->colorBuffer.get());
            Destroy(this->texCoordBuffer.get());
            Destroy(this->animationValuesBuffer.get());
        }

        void DiskGeometry::SetDisks(uint32_t numVertices, const Vec3f* vertices, const Vec3f* normals, const float* radii)
        {
            if (this->numVertices > 0 && numVertices != this->numVertices)
            {
                // Reset vertex attribute arrays if the vertex count has changed
                this->SetColors(nullptr);
                this->SetTexCoords(nullptr);
                this->SetParameterization(nullptr);
            }

            this->numVertices = numVertices;

            if (vertices != nullptr && normals != nullptr)
            {
                optix::Context context = OptiXContext::Get();

                // Positions and squared radii buffer
                if (!this->diskBuffer)
                    this->diskBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, numVertices);
                else
                    this->diskBuffer->setSize(numVertices);

                optix::float4* diskData = reinterpret_cast<optix::float4*>(this->diskBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

                for (uint32_t i = 0; i < numVertices; ++i)
                {
                    const optix::float3 pos = make_float3(vertices[i]);
                    float radius = radii ? radii[i] : 1.0f;
                    diskData[i] = optix::make_float4(pos, radius);
                }

                this->diskBuffer->unmap();

                // Normals buffer (packed)
                if (!this->normalBuffer)
                    this->normalBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numVertices);
                else
                    this->normalBuffer->setSize(numVertices);

                optix::uint* normalData = reinterpret_cast<optix::uint*>(this->normalBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

                for (uint32_t i = 0; i < numVertices; ++i)
                {
                    const optix::float3 normal = make_float3(normals[i]);
                    normalData[i] = packNormal(normal);
                }

                this->normalBuffer->unmap();

                this->instance["positionsAndRadii"]->setInt(diskBuffer->getId());
                this->instance["normals"]->setInt(normalBuffer->getId());
            }
            else
            {
                this->numVertices = 0;

                this->instance["positionsAndRadii"]->setInt(RT_BUFFER_ID_NULL);
                this->instance["normals"]->setInt(RT_BUFFER_ID_NULL);
            }

            this->geometry->setPrimitiveCount(this->numVertices);
            this->MarkDirty();
        }

        void DiskGeometry::SetColors(const Vec4f* vertexColors)
        {
            if (vertexColors)
            {
                optix::Context context = OptiXContext::Get();

                if (!this->colorBuffer)
                    this->colorBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, vertexColors ? this->numVertices : 0);
                else
                    this->colorBuffer->setSize(vertexColors ? this->numVertices : 0);

                memcpy(colorBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertexColors, this->numVertices * sizeof(optix::float4));
                colorBuffer->unmap();

                this->instance["vertexcolors"]->setInt(colorBuffer->getId());
            }
            else
            {
                this->instance["vertexcolors"]->setInt(RT_BUFFER_ID_NULL);
            }
        }

        void DiskGeometry::SetTexCoords(const Vec2f* vertexTexCoords)
        {
            if (vertexTexCoords)
            {
                optix::Context context = OptiXContext::Get();

                if (!texCoordBuffer)
                    texCoordBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, vertexTexCoords ? this->numVertices : 0);
                else
                    this->texCoordBuffer->setSize(vertexTexCoords ? this->numVertices : 0);

                memcpy(texCoordBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertexTexCoords, this->numVertices * sizeof(optix::float2));
                texCoordBuffer->unmap();

                this->instance["texcoords"]->setInt(texCoordBuffer->getId());
            }
            else
            {
                this->instance["texcoords"]->setInt(RT_BUFFER_ID_NULL);
            }
        }

        void DiskGeometry::SetParameterization(const float* vertexParameterization)
        {
            if (vertexParameterization)
            {
                optix::Context context = OptiXContext::Get();

                if (!this->animationValuesBuffer)
                    this->animationValuesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, vertexParameterization ? this->numVertices : 0);
                else
                    this->animationValuesBuffer->setSize(vertexParameterization ? this->numVertices : 0);

                memcpy(animationValuesBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), vertexParameterization, this->numVertices * sizeof(float));
                animationValuesBuffer->unmap();

                this->instance["animationArray"]->setInt(animationValuesBuffer->getId());
            }
            else
            {
                this->instance["animationArray"]->setInt(RT_BUFFER_ID_NULL);
            }
        }

        void DiskGeometry::SetRadius(float radius)
        {
            this->instance["diskRadius"]->setFloat(radius);
            this->MarkDirty();
        }
    }
}
