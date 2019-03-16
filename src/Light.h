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
#include "Pathtracer/Light.h"


namespace VisRTX
{
    namespace Impl
    {
        class Light : public virtual Object, public virtual VisRTX::Light
        {
            friend class Renderer;
            friend class Model;

        public:
            Light();
            virtual ~Light();

        public:
            void SetEnabled(bool enabled) override;
            void SetVisible(bool visible) override;
            void SetColor(const Vec3f& color) override;
            void SetIntensity(float intensity) override;

        public:
            virtual bool Set(::Light& light) const = 0;

            virtual void UpdateGeometry() = 0;
            optix::Material GetLightMaterial() const;

            void MarkDirty(bool updateGeometry);

        protected:
            bool enabled;
            bool visible;
            optix::float3 color;
            float intensity;

            int id;

            bool dirty = true; // used by renderer to update light buffers only when necessary
            VisRTX::Geometry* geometry = nullptr; // added to model for lights visible in scenes
            optix::Buffer lightBuffer;
        };


        class SphericalLight : public VisRTX::SphericalLight, public Light
        {
            friend class Renderer;

        public:
            SphericalLight();
            SphericalLight(const Vec3f& position, const Vec3f& color, float radius);
            ~SphericalLight() = default;

        public:
            void SetPosition(const Vec3f& position) override;
            void SetRadius(float radius) override;

        public:
            bool Set(::Light& light) const override;
            void UpdateGeometry() override;

        private:
            optix::float3 position;
            float radius;
            float falloffScale;
        };


        class DirectionalLight : public VisRTX::DirectionalLight, public Light
        {
            friend class Renderer;

        public:
            DirectionalLight();
            DirectionalLight(const Vec3f& direction, const Vec3f& color);
            ~DirectionalLight() = default;

        public:
            void SetDirection(const Vec3f& direction) override;
            void SetAngularDiameter(float angularDiameter) override;

        public:
            bool Set(::Light& light) const override;
            void UpdateGeometry() override;

        private:
            optix::float3 direction;
            float angularDiameter;
        };


        class QuadLight : public VisRTX::QuadLight, public Light
        {
            friend class Renderer;

        public:
            QuadLight();
            QuadLight(const Vec3f& position, const Vec3f& edge1, const Vec3f& edge2, const Vec3f& color);
            ~QuadLight() = default;

        public:
            void SetRect(const Vec3f& position, const Vec3f& edge1, const Vec3f& edge2) override;
            void SetTwoSided(bool twoSided) override;

        public:
            bool Set(::Light& light) const override;
            void UpdateGeometry() override;

        private:
            optix::float3 position;
            optix::float3 edge1;
            optix::float3 edge2;
            uint32_t twoSided;
            float falloffScale;
        };


        class SpotLight : public VisRTX::SpotLight, public Light
        {
            friend class Renderer;

        public:
            SpotLight();
            SpotLight(const Vec3f& position, const Vec3f& direction, const Vec3f& color, float openingAngle, float penumbraAngle, float radius);
            ~SpotLight() = default;

        public:
            void SetPosition(const Vec3f& position) override;
            void SetDirection(const Vec3f& direction) override;
            void SetOpeningAngle(float openingAngle) override;
            void SetPenumbraAngle(float penumbraAngle) override;
            void SetRadius(float radius) override;

        public:
            bool Set(::Light& light) const override;
            void UpdateGeometry() override;

        private:
            optix::float3 position;
            optix::float3 direction;
            float openingAngle;
            float penumbraAngle;
            float radius;
            float falloffScale;
        };


        class AmbientLight : public VisRTX::AmbientLight, public Light
        {
            friend class Renderer;

        public:
            AmbientLight();
            AmbientLight(const Vec3f& color);
            ~AmbientLight() = default;

        public:
            bool Set(::Light& light) const override;
            void UpdateGeometry() override;
        };


        class HDRILight : public VisRTX::HDRILight, public Light
        {
            friend class Renderer;

        public:
            HDRILight();
            HDRILight(VisRTX::Texture* texture);
            ~HDRILight();

        public:
            void SetTexture(VisRTX::Texture* texture) override;
            void SetDirection(const Vec3f& direction) override;
            void SetUp(const Vec3f& up) override;

        public:
            bool Set(::Light& light) const override;
            void UpdateGeometry() override;

        private:
            VisRTX::Texture* texture = nullptr;
            optix::float3 direction;
            optix::float3 up;            
        };
    }
}

#pragma warning( pop ) // C4250