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

namespace VisRTX
{
    namespace Impl
    {
        class Camera : public virtual Object, public virtual VisRTX::Camera
        {
            friend class Renderer;

        public:
            Camera();
            virtual ~Camera() = default;

        public:
            void SetPosition(const Vec3f& position) override;
            void SetDirection(const Vec3f& direction) override;
            void SetUp(const Vec3f& up) override;

        protected:
            optix::float3 position;
            optix::float3 direction;
            optix::float3 up;

            bool dirty;
        };


        class PerspectiveCamera : public VisRTX::PerspectiveCamera, public Camera
        {
            friend class Renderer;

        public:
            PerspectiveCamera();
            PerspectiveCamera(const Vec3f& position, const Vec3f& direction, const Vec3f& up, float fovy);
            ~PerspectiveCamera() = default;

        public:
            void SetFovY(float fovy) override;
            void SetAspect(float aspect) override;
            void SetFocalDistance(float focalDistance) override;
            void SetApertureRadius(float apertureRadius) override;

        private:
            float fovy;
            float aspect;
            float focalDistance;
            float apertureRadius;
        };


        class OrthographicCamera : public VisRTX::OrthographicCamera, public Camera
        {
            friend class Renderer;

        public:
            OrthographicCamera();
            OrthographicCamera(const Vec3f& position, const Vec3f& direction, const Vec3f& up, float height);
            ~OrthographicCamera() = default;

        public:
            void SetHeight(float height) override;
            void SetAspect(float aspect) override;

        private:            
            float height; // in world coordinates
            float aspect;
        };
    }
}

#pragma warning( pop ) // C4250