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


#include "Camera.h"

namespace VisRTX
{
    namespace Impl
    {
        // ------------------------------------------------------------------------------------
        // Base class
        // ------------------------------------------------------------------------------------
        Camera::Camera()
        {
            this->SetPosition(Vec3f(0.0f, 0.0f, 0.0f));
            this->SetDirection(Vec3f(0.0f, 0.0f, -1.0f));
            this->SetUp(Vec3f(0.0f, 1.0f, 0.0f));
        }

        void Camera::SetPosition(const Vec3f& position)
        {
            if (position.x != this->position.x || position.y != this->position.y || position.z != this->position.z)
            {
                this->position = make_float3(position);
                this->dirty = true;
            }
        }

        void Camera::SetDirection(const Vec3f& direction)
        {
            const optix::float3 dirNorm = optix::normalize(make_float3(direction));
            if (dirNorm.x != this->direction.x || dirNorm.y != this->direction.y || dirNorm.z != this->direction.z)
            {
                this->direction = dirNorm;
                this->dirty = true;
            }
        }

        void Camera::SetUp(const VisRTX::Vec3f& up)
        {
            const optix::float3 upNorm = optix::normalize(make_float3(up));
            if (upNorm.x != this->up.x || upNorm.y != this->up.y || upNorm.z != this->up.z)
            {
                this->up = upNorm;
                this->dirty = true;
            }
        }


        // ------------------------------------------------------------------------------------
        // Perspective camera
        // ------------------------------------------------------------------------------------
        PerspectiveCamera::PerspectiveCamera()
        {
            this->SetFovY(45.0f);
            this->SetAspect(1.0f);
            this->SetFocalDistance(-1.0f);
            this->SetApertureRadius(-1.0f);
        }

        PerspectiveCamera::PerspectiveCamera(const Vec3f& position, const VisRTX::Vec3f& direction, const Vec3f& up, float fovy) : PerspectiveCamera()
        {
            this->SetPosition(position);
            this->SetDirection(direction);
            this->SetUp(up);
            this->SetFovY(fovy);
        }

        void PerspectiveCamera::SetFovY(float fovy)
        {
            if (fovy != this->fovy)
            {
                this->fovy = fovy;
                this->dirty = true;
            }
        }

        void PerspectiveCamera::SetAspect(float aspect)
        {
            if (aspect != this->aspect)
            {
                this->aspect = aspect;
                this->dirty = true;
            }
        }

        void PerspectiveCamera::SetFocalDistance(float focalDistance)
        {
            if (focalDistance != this->focalDistance)
            {
                this->focalDistance = focalDistance;
                this->dirty = true;
            }
        }

        void PerspectiveCamera::SetApertureRadius(float apertureRadius)
        {
            if (apertureRadius != this->apertureRadius)
            {
                this->apertureRadius = apertureRadius;
                this->dirty = true;
            }
        }


        // ------------------------------------------------------------------------------------
        // Orthographic camera
        // ------------------------------------------------------------------------------------
        OrthographicCamera::OrthographicCamera()
        {
            this->SetHeight(1.0f);
            this->SetAspect(1.0f);
        }

        OrthographicCamera::OrthographicCamera(const Vec3f& position, const Vec3f& direction, const Vec3f& up, float height) : OrthographicCamera()
        {
            this->SetPosition(position);
            this->SetDirection(direction);
            this->SetUp(up);
            this->SetHeight(height);
        }
        
        void OrthographicCamera::SetHeight(float height)
        {
            if (height != this->height)
            {
                this->height = height;
                this->dirty = true;
            }
        }

        void OrthographicCamera::SetAspect(float aspect)
        {
            if (aspect != this->aspect)
            {
                this->aspect = aspect;
                this->dirty = true;
            }
        }
    }
}