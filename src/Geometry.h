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

#pragma warning( push )
#pragma warning( disable : 4250 ) // C4250 - 'class1' : inherits 'class2::member' via dominance

#include "VisRTX.h"
#include "OptiX.h"
#include "Object.h"

#include <set>

namespace VisRTX
{
    namespace Impl
    {
        class Geometry : public virtual Object, public virtual VisRTX::Geometry
        {
            friend class Model;
            friend class Renderer;
            friend class Light;

        public:
            Geometry(bool isTriangles);
            virtual ~Geometry();

        public:
            void SetMaterial(VisRTX::Material* material) override;
            void SetAnimatedParameterization(bool enabled, float time, float frequency, float scale) override;            

        public:
            void AddAcceleration(optix::Acceleration acceleration);                        
            void RemoveAcceleration(optix::Acceleration acceleration);
            void MarkDirty();
            void SetMaterial(optix::Material material);

        protected:
            void UpdatePrimitiveMaterialHandles(uint32_t numMaterials, VisRTX::Material** const materials);

        protected:
            optix::GeometryInstance instance;
            optix::Material material;
        
            VisRTX::Material* materialHandle = nullptr;
            std::vector<VisRTX::Material*> primitiveMaterialHandles;

            bool isStatic = true;
            bool isTriangles = false;

            int id;

            struct AccelerationCompare
            {
                bool operator() (const optix::Acceleration& lhs, const optix::Acceleration& rhs) const 
                {
                    return lhs.get() < rhs.get();
                }
            };
            std::set<optix::Acceleration, AccelerationCompare> accelerations;
        };


        class TriangleGeometry : public VisRTX::TriangleGeometry, public Geometry
        {
            friend class Model;

        public:
            TriangleGeometry();
            TriangleGeometry(uint32_t numTriangles, const Vec3ui* triangles, uint32_t numVertices, const Vec3f* vertices, const Vec3f* normals);
            ~TriangleGeometry();

        public:
            void SetTriangles(uint32_t numTriangles, const Vec3ui* triangles, uint32_t numVertices, const Vec3f* vertices, const Vec3f* normals) override;
            void SetColors(const Vec4f* vertexColors) override;
            void SetTexCoords(const Vec2f* vertexTexCoords) override;
            void SetParameterization(const float* vertexParameterization) override;
            void SetMaterials(VisRTX::Material** const triangleMaterials) override;

        private:
#if OPTIX_VERSION_MAJOR >= 6
            optix::GeometryTriangles geometry;
#else
            optix::Geometry geometry;
#endif

            uint32_t numVertices = 0;
            uint32_t numTriangles = 0;

            optix::Buffer vertexBuffer;
            optix::Buffer triangleBuffer;
            optix::Buffer colorBuffer;
            optix::Buffer texCoordBuffer;
            optix::Buffer animationValuesBuffer;            
            optix::Buffer materialBuffer;            
        };


        class SphereGeometry : public VisRTX::SphereGeometry, public Geometry
        {
            friend class Model;

        public:
            SphereGeometry();
            SphereGeometry(uint32_t numVertices, const Vec3f* vertices, const float* radii);
            ~SphereGeometry();

        public:
            void SetSpheres(uint32_t numVertices, const Vec3f* vertices, const float* radii) override;
            void SetColors(const Vec4f* vertexColors) override;
            void SetTexCoords(const Vec2f* vertexTexCoords) override;
            void SetParameterization(const float* vertexParameterization) override;
            void SetRadius(float radius) override;
            void SetSpheresAndColors(uint32_t numVertices, const void* vertexData, uint32_t bytesPerSphere, uint32_t offsetCenter, int32_t offsetRadius, int32_t offsetColorIndex, const Vec4f* colors);
            void SetMaterials(VisRTX::Material** const sphereMaterials) override;

        private:
            optix::Geometry geometry;
            uint32_t numVertices = 0;

            optix::Buffer sphereBuffer;
            optix::Buffer colorBuffer;
            optix::Buffer texCoordBuffer;
            optix::Buffer animationValuesBuffer;
            optix::Buffer materialBuffer;
        };


        class CylinderGeometry : public VisRTX::CylinderGeometry, public Geometry
        {
            friend class Model;

        public:
            CylinderGeometry();
            CylinderGeometry(uint32_t numCylinders, const Vec2ui* cylinders, uint32_t numVertices, const Vec3f* vertices, const float* radii);
            ~CylinderGeometry();

        public:
            void SetCylinders(uint32_t numCylinders, const Vec2ui* cylinders, uint32_t numVertices, const Vec3f* vertices, const float* radii) override;
            void SetColors(const Vec4f* vertexColors) override;
            void SetTexCoords(const Vec2f* vertexTexCoords) override;
            void SetParameterization(const float* vertexParameterization) override;
            void SetRadius(float radius) override;
            void SetCylindersAndColors(uint32_t numCylinders, const void* cylinderData, uint32_t bytesPerCylinder, uint32_t offset_v0, uint32_t offset_v1, int32_t offsetRadius, const Vec4f* cylinderColors) override;
            void SetMaterials(VisRTX::Material** const cylinderMaterials) override;

        private:
            optix::Geometry geometry;
            uint32_t numVertices = 0;
            uint32_t numCylinders = 0;
            
            optix::Buffer vertexBuffer;
            optix::Buffer lineBuffer;
            optix::Buffer colorBuffer;
            optix::Buffer texCoordBuffer;
            optix::Buffer animationValuesBuffer;
            optix::Buffer materialBuffer;
        };


        class DiskGeometry : public VisRTX::DiskGeometry, public Geometry
        {
            friend class Model;

        public:
            DiskGeometry();
            DiskGeometry(uint32_t numVertices, const Vec3f* vertices, const Vec3f* normals, const float* radii);
            ~DiskGeometry();

        public:
            void SetDisks(uint32_t numVertices, const Vec3f* vertices, const Vec3f* normals, const float* radii);
            void SetColors(const Vec4f* vertexColors);
            void SetTexCoords(const Vec2f* vertexTexCoords) override;
            void SetParameterization(const float* vertexParameterization) override;
            void SetRadius(float radius) override;

        private:
            optix::Geometry geometry;
            uint32_t numVertices = 0;

            optix::Buffer diskBuffer; // positions + squared radii
            optix::Buffer normalBuffer; // packed
            optix::Buffer colorBuffer;
            optix::Buffer texCoordBuffer;
            optix::Buffer animationValuesBuffer;
        };       
    }
}

#pragma warning( pop ) // C4250