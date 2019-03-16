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

#include "Model.h"
#include "OptiX.h"
#include "VisRTX.h"
#include "Geometry.h"
#include "Light.h"

namespace VisRTX
{
    namespace Impl
    {
        Model::Model()
        {
            optix::Context context = OptiXContext::Get();

            // Create geometry groups and accelerations
            this->geometryGroupAcceleration = context->createAcceleration("Trbvh", "Bvh");
            this->geometryGroup = context->createGeometryGroup();
            this->geometryGroup->setAcceleration(this->geometryGroupAcceleration);

            this->geometryTrianglesGroupAcceleration = context->createAcceleration("Trbvh", "Bvh");
            this->geometryTrianglesGroup = context->createGeometryGroup();
            this->geometryTrianglesGroup->setAcceleration(this->geometryTrianglesGroupAcceleration);

            this->staticGeometryGroupAcceleration = context->createAcceleration("Trbvh", "Bvh");
            this->staticGeometryGroup = context->createGeometryGroup();
            this->staticGeometryGroup->setAcceleration(this->staticGeometryGroupAcceleration);

            this->staticGeometryTrianglesGroupAcceleration = context->createAcceleration("Trbvh", "Bvh");
            this->staticGeometryTrianglesGroup = context->createGeometryGroup();
            this->staticGeometryTrianglesGroup->setAcceleration(this->staticGeometryTrianglesGroupAcceleration);

            this->superGroupAcceleration = context->createAcceleration("Trbvh", "Bvh");
            this->superGroupAcceleration->setProperty("refit", "1");
            this->superGroup = context->createGroup();
            this->superGroup->setChildCount(4);
            this->superGroup->setChild(0, this->geometryGroup);
            this->superGroup->setChild(1, this->geometryTrianglesGroup);
            this->superGroup->setChild(2, this->staticGeometryGroup);
            this->superGroup->setChild(3, this->staticGeometryTrianglesGroup);
            this->superGroup->setAcceleration(this->superGroupAcceleration);
        }

        Model::~Model()
        {            
            // Release all stored geometries and lights
            for (auto& it : this->addedGeometries)
                if (it.first)
                    it.first->Release();
            
            for (VisRTX::Light* light : this->addedLights)
                if (light)
                    light->Release();

            // Destroy all OptiX instances
            Destroy(this->geometryGroup.get());
            Destroy(this->geometryTrianglesGroup.get());
            Destroy(this->staticGeometryGroup.get());
            Destroy(this->staticGeometryTrianglesGroup.get());
            Destroy(this->superGroup.get());
            Destroy(this->geometryGroupAcceleration.get());
            Destroy(this->geometryTrianglesGroupAcceleration.get());
            Destroy(this->staticGeometryGroupAcceleration.get());
            Destroy(this->staticGeometryTrianglesGroupAcceleration.get());
            Destroy(this->superGroupAcceleration.get());
        }

        void Model::AddGeometry(VisRTX::Geometry* geometry, GeometryFlag flag)
        {
            if (!geometry)
                return;

            // Check if already added
            auto it = this->addedGeometries.find(geometry);
            if (it != this->addedGeometries.end())
                return;

            VisRTX::Impl::Geometry* geo = dynamic_cast<VisRTX::Impl::Geometry*>(geometry);

            if (flag == GeometryFlag::STATIC)
            {
                if (geo->isTriangles)
                {
                    this->staticGeometryTrianglesGroup->addChild(geo->instance);
                    this->staticGeometryTrianglesGroupAcceleration->markDirty();                    
                    
                    geo->AddAcceleration(this->staticGeometryTrianglesGroupAcceleration);
                    this->addedGeometries[geometry] = this->staticGeometryTrianglesGroup;
                }
                else
                {
                    this->staticGeometryGroup->addChild(geo->instance);
                    this->staticGeometryGroupAcceleration->markDirty();

                    geo->AddAcceleration(this->staticGeometryGroupAcceleration);
                    this->addedGeometries[geometry] = this->staticGeometryGroup;
                }
            }
            else
            {
                if (geo->isTriangles)
                {
                    this->geometryTrianglesGroup->addChild(geo->instance);
                    this->geometryTrianglesGroupAcceleration->markDirty();

                    geo->AddAcceleration(this->geometryTrianglesGroupAcceleration);
                    this->addedGeometries[geometry] = this->geometryTrianglesGroup;
                }
                else
                {
                    this->geometryGroup->addChild(geo->instance);
                    this->geometryGroupAcceleration->markDirty();

                    geo->AddAcceleration(this->geometryGroupAcceleration);
                    this->addedGeometries[geometry] = this->geometryGroup;
                }
            }

            this->superGroupAcceleration->markDirty();
            geo->AddAcceleration(this->superGroupAcceleration);

            // Add to picking map
            this->pickGeometries[geo->id] = geometry;

            // Increase ref count
            geometry->Retain();
        }

        void Model::RemoveGeometry(VisRTX::Geometry* geometry)
        {
            if (!geometry)
                return;

            VisRTX::Impl::Geometry* geo = dynamic_cast<VisRTX::Impl::Geometry*>(geometry);

            auto it = this->addedGeometries.find(geometry);
            if (it != this->addedGeometries.end())
            {
                optix::GeometryGroup group = it->second;
                optix::Acceleration acceleration = group->getAcceleration();
                this->addedGeometries.erase(it);

                geo->RemoveAcceleration(acceleration);
                
                group->removeChild(geo->instance);
                group->getAcceleration()->markDirty();
                this->superGroupAcceleration->markDirty();

                // Remove from picking map
                this->pickGeometries.erase(geo->id);

                // Decrease ref count
                geometry->Release();
            }
        }

        void Model::UpdateLightGeometries(std::set<VisRTX::Light*>* lights)
        {
            // Clear all lights from this model
            if (nullptr == lights)
            {
                for (VisRTX::Light* light : this->addedLights)
                {
                    VisRTX::Impl::Light* l = dynamic_cast<VisRTX::Impl::Light*>(light);
                    this->RemoveGeometry(l->geometry);

                    light->Release();
                }

                this->addedLights.clear();
            }

            // Add/remove lights to match given set
            else
            {
                // Remove lights from model NOT in parameter set
                // Remove lights from model that are DISABLED
                for (auto it = this->addedLights.begin(); it != this->addedLights.end();)
                {
                    VisRTX::Impl::Light* light = dynamic_cast<VisRTX::Impl::Light*>(*it);
                  
                    const bool remove = (!light->enabled) || (lights->find(*it) == lights->end());
                    if (remove)
                    {
                        this->RemoveGeometry(light->geometry);
                        it = this->addedLights.erase(it);

                        light->Release();
                    }
                    else
                    {
                        ++it;
                    }
                }

                // Add ENABLED lights from parameter set NOT in model set                
                for (VisRTX::Light* other : *lights)
                {
                    VisRTX::Impl::Light* light = dynamic_cast<VisRTX::Impl::Light*>(other);

                    const bool add = (light->enabled) && (this->addedLights.find(other) == this->addedLights.end());
                    if (add)
                    {
                        this->AddGeometry(light->geometry, VisRTX::GeometryFlag::DYNAMIC);
                        this->addedLights.insert(other);

                        light->Retain();
                    }
                }
            }
        }
    }
}
