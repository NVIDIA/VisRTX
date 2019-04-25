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

#include "Light.h"
#include "Texture.h"
#include "ProgramLoader.h"
#include "Geometry.h"

#include <algorithm>


namespace VisRTX
{
    namespace Impl
    {
        // ------------------------------------------------------------------------------------
        // Base class
        // ------------------------------------------------------------------------------------
        Light::Light()
        {
            this->SetEnabled(true);
            this->SetVisible(true);
            this->SetColor(Vec3f(1.0f, 1.0f, 1.0f));
            this->SetIntensity(1.0f);

            static int lightId = 0;
            this->id = ++lightId;
        }

        Light::~Light()
        {
            if (this->geometry)
                this->geometry->Release();

            Destroy(this->lightBuffer.get());
        }

        void Light::SetEnabled(bool enabled)
        {
            if (enabled != this->enabled)
            {
                this->enabled = enabled;
                this->MarkDirty(false);
            }
        }

        void Light::SetVisible(bool visible)
        {
            if (visible != this->visible)
            {
                this->visible = visible;
                this->MarkDirty(false);
            }
        }

        void Light::SetColor(const Vec3f& color)
        {
            if (color.x != this->color.x || color.y != this->color.y || color.z != this->color.z)
            {
                this->color = make_float3(color);
                this->MarkDirty(false);
            }
        }

        void Light::SetIntensity(float intensity)
        {
            if (intensity != this->intensity)
            {
                this->intensity = intensity;
                this->MarkDirty(false);
            }
        }

        void Light::MarkDirty(bool updateGeometry)
        {
            this->dirty = true;

            if (updateGeometry)
                this->UpdateGeometry();

            if (this->geometry)
            {               
                if (!this->lightBuffer)
                {
                    this->lightBuffer = OptiXContext::Get()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
                    this->lightBuffer->setElementSize(sizeof(::Light));
                    this->lightBuffer->setSize(1);

                    dynamic_cast<VisRTX::Impl::Geometry*>(this->geometry)->instance["light"]->setBuffer(this->lightBuffer);
                }

                ::Light* light = static_cast<::Light*>(this->lightBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
                this->Set(*light);
                this->lightBuffer->unmap();                
            }
        }

        optix::Material Light::GetLightMaterial() const
        {
            static optix::Material lightMaterial;
            
            if (!lightMaterial)
            {
                lightMaterial = OptiXContext::Get()->createMaterial();

                ProgramLoader& loader = ProgramLoader::Get();
                lightMaterial->setClosestHitProgram(RADIANCE_RAY_TYPE, loader.lightClosestHitProgram);
                lightMaterial->setAnyHitProgram(RADIANCE_RAY_TYPE, loader.lightAnyHitProgram);
                lightMaterial->setAnyHitProgram(OCCLUSION_RAY_TYPE, loader.lightAnyHitOcclusionProgram);
                lightMaterial->setClosestHitProgram(PICK_RAY_TYPE, loader.lightClosestHitPickProgram);                
            }

            return lightMaterial;
        }


        // ------------------------------------------------------------------------------------
        // Spherical light
        // ------------------------------------------------------------------------------------
        SphericalLight::SphericalLight()
        {
            this->SetPosition(Vec3f(0.0f, 0.0f, 0.0f));
            this->SetRadius(0.0f);
        }

        SphericalLight::SphericalLight(const Vec3f& position, const Vec3f& color, float radius) : SphericalLight()
        {
            this->SetPosition(position);
            this->SetColor(color);
            this->SetRadius(radius);
        }

        void SphericalLight::SetPosition(const Vec3f& position)
        {
            if (position.x != this->position.x || position.y != this->position.y || position.z != this->position.z)
            {
                this->position = make_float3(position);
                this->MarkDirty(true);
            }
        }

        void SphericalLight::SetRadius(float radius)
        {
            if (radius != this->radius)
            {
                this->radius = radius;
                this->MarkDirty(true);
            }
        }

        bool SphericalLight::Set(::Light& light) const
        {
            if (!this->enabled)
                return false;

            light.id = this->id;

            light.color = this->intensity * this->color;
            light.visible = this->visible ? 1 : 0;

            light.type = ::Light::POSITIONAL;
            light.pos = this->position;
            light.radius = this->radius;
            light.twoSided = 1;

            return true;
        }

        void SphericalLight::UpdateGeometry()
        {
            VisRTX::Impl::SphereGeometry* sphere;

            if (!this->geometry)
            {
                sphere = new VisRTX::Impl::SphereGeometry();
                sphere->SetMaterial(this->GetLightMaterial(), true);
                this->geometry = sphere;
            }
            else
            {
                sphere = dynamic_cast<VisRTX::Impl::SphereGeometry*>(this->geometry);
            }            

            Vec3f center(this->position.x, this->position.y, this->position.z);
            sphere->SetSpheres(1, &center, nullptr);
            sphere->SetRadius(this->radius);
        }


        // ------------------------------------------------------------------------------------
        // Directional light
        // ------------------------------------------------------------------------------------
        DirectionalLight::DirectionalLight()
        {
            this->SetDirection(Vec3f(0.0f, -1.0f, 0.0f));
            this->SetAngularDiameter(0.0f);
        }

        DirectionalLight::DirectionalLight(const Vec3f& direction, const Vec3f& color) : DirectionalLight()
        {
            this->SetDirection(direction);
            this->SetColor(color);
        }

        void DirectionalLight::SetDirection(const Vec3f& direction)
        {
            const optix::float3 normDir = optix::normalize(make_float3(direction));

            if (normDir.x != this->direction.x || normDir.y != this->direction.y || normDir.z != this->direction.z)
            {
                this->direction = normDir;
                this->MarkDirty(false);
            }
        }

        void DirectionalLight::SetAngularDiameter(float angularDiameter)
        {
            if (angularDiameter != this->angularDiameter)
            {
                this->angularDiameter = angularDiameter;
                this->MarkDirty(false);
            }
        }

        bool DirectionalLight::Set(::Light& light) const
        {
            if (!this->enabled)
                return false;

            light.id = this->id;

            light.color = this->intensity * this->color;
            light.visible = this->visible ? 1 : 0;

            light.type = ::Light::DIRECTIONAL;
            light.dir = this->direction;
            light.angularDiameter = 0.5f * this->angularDiameter * 0.01745329251f;

            const float cosAngle = cos(light.angularDiameter);
            light.pdf = (this->angularDiameter > 0.0f) ? (1.0f / (2.0f * M_PIf * (1.0f - cosAngle))) : PDF_DIRAC;
          
            return true;
        }

        void DirectionalLight::UpdateGeometry()
        {
            // No geometry
        }


        // ------------------------------------------------------------------------------------
        // Quad light
        // ------------------------------------------------------------------------------------
        QuadLight::QuadLight()
        {
            this->SetRect(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(1.0f, 0.0f, 0.0f), Vec3f(0.0f, 1.0f, 0.0f));
            this->SetTwoSided(false);
        }

        QuadLight::QuadLight(const Vec3f& position, const Vec3f& edge1, const Vec3f& edge2, const Vec3f& color) : QuadLight()
        {
            this->SetRect(position, edge1, edge2);
            this->SetColor(color);
        }

        void QuadLight::SetRect(const Vec3f& position, const Vec3f& edge1, const Vec3f& edge2)
        {
            if (position.x != this->position.x || position.y != this->position.y || position.z != this->position.z
                || edge1.x != this->edge1.x || edge1.y != this->edge1.y || edge1.z != this->edge1.z
                || edge2.x != this->edge2.x || edge2.y != this->edge2.y || edge2.z != this->edge2.z)
            {
                this->position = make_float3(position);
                this->edge1 = make_float3(edge1);
                this->edge2 = make_float3(edge2);
                this->MarkDirty(true);
            }
        }

        void QuadLight::SetTwoSided(bool twoSided)
        {
            const uint32_t ts = twoSided ? 1 : 0;
            
            if (ts != this->twoSided)
            {
                this->twoSided = ts;                
                this->MarkDirty(false);
            }
        }

        bool QuadLight::Set(::Light& light) const
        {
            if (!this->enabled)
                return false;

            light.id = this->id;

            light.color = this->intensity * this->color;
            light.visible = this->visible ? 1 : 0;

            light.type = ::Light::QUAD;
            light.pos = this->position;
            light.edge1 = this->edge1;
            light.edge2 = this->edge2;
            light.twoSided = this->twoSided;

            const float area = optix::length(optix::cross(this->edge1, this->edge2));
            light.pdf = area > 0.0f ? (1.0f / area) : PDF_DIRAC;

            return true;
        }

        void QuadLight::UpdateGeometry()
        {
            VisRTX::Impl::TriangleGeometry* triangles;

            if (!this->geometry)
            {
                triangles = new VisRTX::Impl::TriangleGeometry();
                triangles->SetMaterial(this->GetLightMaterial(), true);
                this->geometry = triangles;
            }
            else
            {
                triangles = dynamic_cast<VisRTX::Impl::TriangleGeometry*>(this->geometry);
            }

            Vec3f vertices[4] = 
            {
                Vec3f(this->position.x, this->position.y, this->position.z),
                Vec3f(this->position.x + this->edge1.x, this->position.y + this->edge1.y, this->position.z + this->edge1.z),
                Vec3f(this->position.x + this->edge1.x + this->edge2.x, this->position.y + this->edge1.y + this->edge2.y, this->position.z + this->edge1.z + this->edge2.z),
                Vec3f(this->position.x + this->edge2.x, this->position.y + this->edge2.y, this->position.z + this->edge2.z)
            };

            Vec3ui indices[2] = 
            {
                Vec3ui(0, 1, 2),
                Vec3ui(0, 2, 3)
            };

            triangles->SetTriangles(2, indices, 4, vertices, nullptr);
        }


        // ------------------------------------------------------------------------------------
        // Spot light
        // ------------------------------------------------------------------------------------
        SpotLight::SpotLight()
        {
            this->SetPosition(Vec3f(0.0f, 0.0f, 0.0f));
            this->SetDirection(Vec3f(0.0f, 0.0f, -1.0f));
            this->SetOpeningAngle(45.0f);
            this->SetPenumbraAngle(5.0f);
            this->SetRadius(0.0f);
        }

        SpotLight::SpotLight(const Vec3f& position, const Vec3f& direction, const Vec3f& color, float openingAngle, float penumbraAngle, float radius) : SpotLight()
        {
            this->SetPosition(position);
            this->SetDirection(direction);
            this->SetColor(color);
            this->SetOpeningAngle(openingAngle);
            this->SetPenumbraAngle(penumbraAngle);
            this->SetRadius(radius);
        }

        void SpotLight::SetPosition(const Vec3f& position)
        {
            if (position.x != this->position.x || position.y != this->position.y || position.z != this->position.z)
            {
                this->position = make_float3(position);
                this->MarkDirty(true);
            }
        }

        void SpotLight::SetDirection(const Vec3f& direction)
        {
            const optix::float3 normDir = optix::normalize(make_float3(direction));

            if (normDir.x != this->direction.x || normDir.y != this->direction.y || normDir.z != this->direction.z)
            {
                this->direction = normDir;
                this->MarkDirty(true);
            }
        }

        void SpotLight::SetOpeningAngle(float openingAngle)
        {
            if (openingAngle != this->openingAngle)
            {
                this->openingAngle = openingAngle;
                this->MarkDirty(false);
            }
        }

        void SpotLight::SetPenumbraAngle(float penumbraAngle)
        {
            if (penumbraAngle != this->penumbraAngle)
            {
                this->penumbraAngle = penumbraAngle;
                this->MarkDirty(false);
            }
        }

        void SpotLight::SetRadius(float radius)
        {
            if (radius != this->radius)
            {
                this->radius = radius;
                this->MarkDirty(true);
            }
        }

        bool SpotLight::Set(::Light& light) const
        {
            if (!this->enabled)
                return false;

            light.id = this->id;

            light.color = this->intensity * this->color;
            light.visible = this->visible ? 1 : 0;

            light.type = ::Light::SPOT;
            light.pos = this->position;
            light.dir = this->direction;

            light.radius = this->radius;
            light.twoSided = 0;

            light.outerAngle = 0.5f * std::min(this->openingAngle, 180.0f) * 0.01745329251f;

            if (this->penumbraAngle < 0.5f * this->openingAngle)
                light.innerAngle = 0.5f * (this->openingAngle - 2.0f * this->penumbraAngle) * 0.01745329251f;
            else
                light.innerAngle = 0.0f;

            const float area = M_PIf * this->radius * this->radius;
            light.pdf = area > 0.0f ? (1.0f / area) : PDF_DIRAC;

            return true;
        }

        void SpotLight::UpdateGeometry()
        {
            VisRTX::Impl::DiskGeometry* disks;

            if (!this->geometry)
            {
                disks = new VisRTX::Impl::DiskGeometry();
                disks->SetMaterial(this->GetLightMaterial(), true);
                this->geometry = disks;
            }
            else
            {
                disks = dynamic_cast<VisRTX::Impl::DiskGeometry*>(this->geometry);
            }           

            Vec3f center(this->position.x, this->position.y, this->position.z);
            Vec3f normal(this->direction.x, this->direction.y, this->direction.z);
            disks->SetDisks(1, &center, &normal, nullptr);
            disks->SetRadius(this->radius);
        }


        // ------------------------------------------------------------------------------------
        // Ambient light
        // ------------------------------------------------------------------------------------
        AmbientLight::AmbientLight()
        {}

        AmbientLight::AmbientLight(const Vec3f& color) : AmbientLight()
        {
            this->SetColor(color);
        }

        bool AmbientLight::Set(::Light& light) const
        {
            if (!this->enabled)
                return false;

            light.id = this->id;

            light.color = this->intensity * this->color;
            light.visible = this->visible ? 1 : 0;

            light.type = ::Light::AMBIENT;

            return true;
        }

        void AmbientLight::UpdateGeometry()
        {
            // No geometry
        }


        // ------------------------------------------------------------------------------------
        // HDRI light
        // ------------------------------------------------------------------------------------
        HDRILight::HDRILight()
        {
            this->SetDirection(Vec3f(0.0f, -1.0f, 0.0f));
            this->SetUp(Vec3f(0.0f, 1.0f, 0.0f));
        }

        HDRILight::HDRILight(VisRTX::Texture* texture) : HDRILight()
        {
            this->SetTexture(texture);
        }

        HDRILight::~HDRILight()
        {
            if (this->texture)
                this->texture->Release();
        }

        void HDRILight::SetTexture(VisRTX::Texture* texture)
        {
            if (texture != this->texture)
            {
                if (this->texture)
                    this->texture->Release();

                this->texture = texture;

                if (this->texture)
                    this->texture->Retain();

                this->MarkDirty(false);
            }            
        }

        void HDRILight::SetDirection(const Vec3f& direction)
        {
            const optix::float3 normDir = optix::normalize(make_float3(direction));

            if (normDir.x != this->direction.x || normDir.y != this->direction.y || normDir.z != this->direction.z)
            {
                this->direction = normDir;
                this->MarkDirty(false);
            }
        }    

        void HDRILight::SetUp(const Vec3f& up)
        {
            const optix::float3 normUp = optix::normalize(make_float3(up));

            if (normUp.x != this->up.x || normUp.y != this->up.y || normUp.z != this->up.z)
            {
                this->up = normUp;
                this->MarkDirty(false);
            }
        }

        bool HDRILight::Set(::Light& light) const
        {
            if (!this->enabled)
                return false;

            light.id = this->id;

            light.color = this->intensity * this->color;
            light.visible = this->visible ? 1 : 0;

            light.type = ::Light::HDRI;

            light.texture = this->texture ? dynamic_cast<VisRTX::Impl::Texture*>(this->texture)->sampler->getId() : RT_TEXTURE_ID_NULL;
            light.dir = this->direction;
            light.up = this->up;

            return true;
        }

        void HDRILight::UpdateGeometry()
        {
            // No geometry
        }
    }
}