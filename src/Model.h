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

#include "OptiX.h"
#include "VisRTX.h"
#include "Object.h"

#include <map>
#include <set>

namespace VisRTX
{
    namespace Impl
    {
        class Model :public virtual Object, public VisRTX::Model
        {
            friend class Renderer;

        public:
            Model();
            ~Model();

        public:
            void AddGeometry(VisRTX::Geometry* geometry, GeometryFlag flag) override;
            void RemoveGeometry(VisRTX::Geometry* geometry) override;
            
        public:
            void UpdateLightGeometries(std::set<VisRTX::Light*>* lights);

        private:
            optix::GeometryGroup geometryGroup;
            optix::GeometryGroup geometryTrianglesGroup;
            optix::GeometryGroup staticGeometryGroup;
            optix::GeometryGroup staticGeometryTrianglesGroup;
            optix::Group         superGroup;
            optix::Acceleration  geometryGroupAcceleration;
            optix::Acceleration  geometryTrianglesGroupAcceleration;
            optix::Acceleration  staticGeometryGroupAcceleration;
            optix::Acceleration  staticGeometryTrianglesGroupAcceleration;
            optix::Acceleration  superGroupAcceleration;

            std::map<VisRTX::Geometry*, optix::GeometryGroup> addedGeometries;
            std::set<VisRTX::Light*> addedLights;

            std::map<int, VisRTX::Geometry*> pickGeometries;
        };
    }
}

#pragma warning( pop )