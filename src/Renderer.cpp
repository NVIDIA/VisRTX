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

#include "Renderer.h"
#include "FrameBuffer.h"
#include "Model.h"
#include "ProgramLoader.h"
#include "Camera.h"
#include "Texture.h"
#include "Material.h"
#include "Light.h"
#include "Geometry.h"
#include "Config.h"

#include "Pathtracer/Light.h"
#include "Pathtracer/Pick.h"

#include <fstream>
#include <algorithm>
#include <cstring>


#define UPDATE_LAUNCH_PARAMETER(src, dst) if (src != dst) { dst = src; this->launchParametersDirty = true; }


namespace VisRTX
{
    namespace Impl
    {
        Renderer::Renderer()
        {
            // Load programs
            ProgramLoader& loader = ProgramLoader::Get();
            this->rayGenProgram = loader.LoadPathtracerProgram("RayGen");
            this->rayGenPickProgram = loader.LoadPathtracerProgram("RayGenPick");
            this->missProgram = loader.LoadPathtracerProgram("Miss");
            this->missPickProgram = loader.LoadPathtracerProgram("MissPick");
            this->bufferCastProgram = loader.LoadPathtracerProgram("BufferCast");

            // Defaults
            this->SetSamplesPerPixel(1);
            this->SetAlphaCutoff(1e-3f);
            this->SetWriteBackground(true);
            this->SetClippingPlanes(0, nullptr);

            uint32_t minBounces = 4;
            uint32_t maxBounces = 8;
            float epsilon = 1e-3f;
            bool toneMapping = true;
            DenoiserType denoiser = DenoiserType::NONE;
            float clampDirect = 0.0f;
            float clampIndirect = 0.0f;
            bool sampleAllLights = true;

            // Env var overrides
            if (const char* str = std::getenv("VISRTX_MIN_BOUNCES"))
            {
                minBounces = (uint32_t)atoi(str);
                this->minBouncesFixed = true;
            }

            if (const char* str = std::getenv("VISRTX_MAX_BOUNCES"))
            {
                maxBounces = (uint32_t)atoi(str);
                this->maxBouncesFixed = true;
            }

            if (const char* str = std::getenv("VISRTX_EPSILON"))
            {
                epsilon = (float)atof(str);
                this->epsilonFixed = true;
            }

            if (const char* str = std::getenv("VISRTX_TONE_MAPPING"))
            {
                toneMapping = atoi(str) > 0;
                this->toneMappingFixed = true;
            }

            if (const char* str = std::getenv("VISRTX_DENOISER"))
            {
                denoiser = (atoi(str) > 0) ? DenoiserType::AI : DenoiserType::NONE;
                this->denoiserFixed = true;
            }

            if (const char* str = std::getenv("VISRTX_CLAMP_DIRECT"))
            {
                clampDirect = (float)atof(str);
                this->clampDirectFixed = true;
            }

            if (const char* str = std::getenv("VISRTX_CLAMP_INDIRECT"))
            {
                clampIndirect = (float)atof(str);
                this->clampIndirectFixed = true;
            }

            if (const char* str = std::getenv("VISRTX_SAMPLE_ALL_LIGHTS"))
            {
                sampleAllLights = atoi(str) > 0;
                this->sampleAllLightsFixed = true;
            }

            this->ignoreOverrides = true;
            this->SetNumBounces(minBounces, maxBounces);
            this->SetEpsilon(epsilon);
            this->SetToneMapping(toneMapping, 2.2f, Vec3f(1.0f, 1.0f, 1.0f), 1.0f, 0.8f, 0.2f, 1.2f, 0.8f);
            this->SetDenoiser(denoiser);
            this->SetFireflyClamping(clampDirect, clampIndirect);
            this->SetSampleAllLights(sampleAllLights);
            this->ignoreOverrides = false;



            // Debug warnings
#ifdef TEST_DIRECT_ONLY
            std::cout << "*** Warning: Testing mode! Brute force (light evaluation) only ***" << std::endl;
#endif
#ifdef TEST_NEE_ONLY
            std::cout << "*** Warning: Testing mode! Next event estimation (light sampling) only ***" << std::endl;
#endif
#ifdef TEST_MIS
            std::cout << "*** Warning: Testing mode! Multiple importance sampling ***" << std::endl;
#endif
        }

        Renderer::~Renderer()
        {
            if (this->model)
                this->model->Release();

            if (this->camera)
                this->camera->Release();

            for (VisRTX::Light* light : this->lights)
                if (light)
                    light->Release();

            Destroy(rayGenProgram.get());
            Destroy(rayGenPickProgram.get());
            Destroy(missProgram.get());
            Destroy(missPickProgram.get());
            Destroy(bufferCastProgram.get());
            Destroy(basicMaterialParametersBuffer.get());
            Destroy(mdlMaterialParametersBuffer.get());
            Destroy(directLightsBuffer.get());
            Destroy(missLightsBuffer.get());
            Destroy(clippingPlanesBuffer.get());
            Destroy(pickBuffer.get());
            Destroy(launchParametersBuffer.get());
        }

        void Renderer::Render(VisRTX::FrameBuffer* frameBuffer)
        {
            if (!frameBuffer)
                return;

            // Initialization prepass
            if (!this->Init())
                return;

            optix::Context context = OptiXContext::Get();


            // Framebuffer
            VisRTX::Impl::FrameBuffer* fb = dynamic_cast<VisRTX::Impl::FrameBuffer*>(frameBuffer);
            const int accumulationBufferId = fb->accumulationBuffer->getId();
            const int frameBufferId = fb->frameBuffer->getId();
            const int ucharFrameBufferId = fb->ucharFrameBuffer->getId();
            const int depthBufferId = fb->depthBuffer->getId();

            const int useAIDenoiser = ((this->denoiser == DenoiserType::AI) && fb->denoiserStage) ? 1 : 0;
            const int writeFrameBuffer = (fb->format == FrameBufferFormat::RGBA32F || useAIDenoiser) ? 1 : 0;
            const int writeUcharFrameBuffer = (fb->format == FrameBufferFormat::RGBA8 && !useAIDenoiser) ? 1 : 0;

            UPDATE_LAUNCH_PARAMETER((int) fb->width, this->launchParameters.width);
            UPDATE_LAUNCH_PARAMETER((int) fb->height, this->launchParameters.height);

            UPDATE_LAUNCH_PARAMETER(accumulationBufferId, this->launchParameters.accumulationBuffer);
            UPDATE_LAUNCH_PARAMETER(frameBufferId, this->launchParameters.frameBuffer);
            UPDATE_LAUNCH_PARAMETER(ucharFrameBufferId, this->launchParameters.ucharFrameBuffer);
            UPDATE_LAUNCH_PARAMETER(depthBufferId, this->launchParameters.depthBuffer);

            UPDATE_LAUNCH_PARAMETER(fb->depthClipMin, this->launchParameters.clipMin);
            UPDATE_LAUNCH_PARAMETER(fb->depthClipMax, this->launchParameters.clipMax);
            UPDATE_LAUNCH_PARAMETER(fb->depthClipDiv, this->launchParameters.clipDiv);            

            UPDATE_LAUNCH_PARAMETER(useAIDenoiser, this->launchParameters.useAIDenoiser);
            UPDATE_LAUNCH_PARAMETER(writeFrameBuffer, this->launchParameters.writeFrameBuffer);
            UPDATE_LAUNCH_PARAMETER(writeUcharFrameBuffer, this->launchParameters.writeUcharFrameBuffer);


            // Material buffers
            if (BasicMaterial::parametersDirty)
            {
                this->basicMaterialParametersBuffer->setSize(BasicMaterial::parameters.GetNumElements());
                const uint32_t numBytes = BasicMaterial::parameters.GetNumBytes();
                if (numBytes > 0)
                {
                    memcpy(this->basicMaterialParametersBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), BasicMaterial::parameters.GetData(), numBytes);
                    this->basicMaterialParametersBuffer->unmap();
                }
                BasicMaterial::parametersDirty = false;
            }

            if (MDLMaterial::parametersDirty)
            {
                this->mdlMaterialParametersBuffer->setSize(MDLMaterial::parameters.GetNumElements());
                const uint32_t numBytes = MDLMaterial::parameters.GetNumBytes();
                if (numBytes > 0)
                {
                    memcpy(this->mdlMaterialParametersBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), MDLMaterial::parameters.GetData(), numBytes);
                    this->mdlMaterialParametersBuffer->unmap();
                }
                MDLMaterial::parametersDirty = false;
            }


            // Camera
            VisRTX::Impl::Camera* camera = dynamic_cast<VisRTX::Impl::Camera*>(this->camera);
            if (camera->dirty)
            {
                const optix::float3 cam_W = optix::normalize(camera->direction);
                const optix::float3 cam_U = optix::normalize(optix::cross(camera->direction, camera->up));
                const optix::float3 cam_V = optix::normalize(optix::cross(cam_U, cam_W));				

                if (this->camera->GetType() == CameraType::PERSPECTIVE)
                {
                    VisRTX::Impl::PerspectiveCamera* perspectiveCam = dynamic_cast<VisRTX::Impl::PerspectiveCamera*>(this->camera);

                    const float vlen = tanf(0.5f * perspectiveCam->fovy * M_PIf / 180.0f);
                    const float ulen = vlen * perspectiveCam->aspect;

                    this->launchParameters.cameraType = PERSPECTIVE_CAMERA;
                    this->launchParameters.pos = perspectiveCam->position;
                    this->launchParameters.U = cam_U * ulen;
                    this->launchParameters.V = cam_V * vlen;
                    this->launchParameters.W = cam_W;
                    this->launchParameters.focalDistance = perspectiveCam->focalDistance;
                    this->launchParameters.apertureRadius = perspectiveCam->apertureRadius;
                }
                else if (this->camera->GetType() == CameraType::ORTHOGRAPHIC)
                {
                    VisRTX::Impl::OrthographicCamera* orthoCam = dynamic_cast<VisRTX::Impl::OrthographicCamera*>(this->camera);

                    this->launchParameters.cameraType = ORTHOGRAPHIC_CAMERA;
                    this->launchParameters.pos = orthoCam->position;
                    this->launchParameters.U = cam_U;
                    this->launchParameters.V = cam_V;
                    this->launchParameters.W = cam_W;
                    this->launchParameters.orthoWidth = orthoCam->height * orthoCam->aspect;
                    this->launchParameters.orthoHeight = orthoCam->height;
                }

				this->launchParameters.imageBegin = camera->imageBegin;
				this->launchParameters.imageSize = camera->imageEnd - camera->imageBegin;

                camera->dirty = false;
                this->launchParametersDirty = true;
            }


            // Launch context
            for (uint32_t i = 0; i < this->samplesPerPixel; ++i)
            {
                UPDATE_LAUNCH_PARAMETER((int) fb->frameNumber, this->launchParameters.frameNumber);

                if (this->launchParametersDirty)
                {
                    memcpy(this->launchParametersBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), &this->launchParameters, sizeof(LaunchParameters));
                    this->launchParametersBuffer->unmap();

                    this->launchParametersDirty = false;
                }

                // Validate only after all nodes have completed the prepass so all context subcomponents have been initialized
                if (!this->contextValidated)
                {
                    try
                    {
                        context->validate();
                    }
                    catch (optix::Exception& e)
                    {
                        throw VisRTX::Exception(Error::UNKNOWN_ERROR, e.getErrorString().c_str());
                    }

                    this->contextValidated = true;
                }

                // Render frame
#ifdef VISRTX_USE_DEBUG_EXCEPTIONS
                try
                {
#endif
                    const bool lastSample = (i == samplesPerPixel - 1);

                    if (lastSample && useAIDenoiser && fb->frameNumber < this->blendDelay + this->blendDuration)
                    {
                        float blend = 0.0f;
                        if (fb->frameNumber >= this->blendDelay)
                            blend = (float)(fb->frameNumber - this->blendDelay) / (float)this->blendDuration;

                        fb->denoiserBlend->setFloat(blend);
                        fb->denoiserCommandList->execute();
                    }
                    else
                    {
                        context->launch(RENDER_ENTRY_POINT, fb->width, fb->height);
                    }
#ifdef VISRTX_USE_DEBUG_EXCEPTIONS
                }
                catch (optix::Exception& e)
                {
                    throw VisRTX::Exception(Error::UNKNOWN_ERROR, e.getErrorString().c_str());
                }
#endif

                // Increase frame number
                fb->frameNumber++;
            }
        }


        /*
         * Common per-frame initialization used by both render and pick.
         */
        bool Renderer::Init()
        {
            if (!this->model)
                return false;

            if (!this->camera)
                return false;

            optix::Context context = OptiXContext::Get();

            /*
             * On-demand initialization
             */
            if (!this->initialized)
            {
#ifdef VISRTX_USE_DEBUG_EXCEPTIONS
                context->setExceptionProgram(RENDER_ENTRY_POINT, ProgramLoader::Get().exceptionProgram);
#endif

                // Launch parameters
                this->launchParametersBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
                this->launchParametersBuffer->setElementSize(sizeof(LaunchParameters));
                this->launchParametersBuffer->setSize(1);
                context["launchParameters"]->setBuffer(this->launchParametersBuffer);
                //this->rayGenProgram["launchParameters"]->setBuffer(this->launchParametersBuffer);
                //this->rayGenPickProgram["launchParameters"]->setBuffer(this->launchParametersBuffer);
                //this->bufferCastProgram["launchParameters"]->setBuffer(this->launchParametersBuffer);
                //this->missProgram["launchParameters"]->setBuffer(this->launchParametersBuffer);
                //this->missPickProgram["launchParameters"]->setBuffer(this->launchParametersBuffer);


                // Material buffers (global buffers shared across renderers/materials)
                this->basicMaterialParametersBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
                this->basicMaterialParametersBuffer->setElementSize(sizeof(BasicMaterialParameters));
                this->basicMaterialParametersBuffer->setSize(0);
                context["basicMaterialParameters"]->setBuffer(this->basicMaterialParametersBuffer);

                this->mdlMaterialParametersBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
                this->mdlMaterialParametersBuffer->setElementSize(sizeof(MDLMaterialParameters));
                this->mdlMaterialParametersBuffer->setSize(0);
                context["mdlMaterialParameters"]->setBuffer(this->mdlMaterialParametersBuffer);


                // Light buffers
                this->directLightsBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
                this->directLightsBuffer->setElementSize(sizeof(::Light));
                this->directLightsBuffer->setSize(0);

                this->missLightsBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
                this->missLightsBuffer->setElementSize(sizeof(::Light));
                this->missLightsBuffer->setSize(0);

                this->rayGenProgram["lights"]->setBuffer(this->directLightsBuffer);
                this->missProgram["lights"]->setBuffer(this->missLightsBuffer);
                this->missPickProgram["lights"]->setBuffer(this->missLightsBuffer);


                // Clipping planes buffer
                this->clippingPlanesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
                this->clippingPlanesBuffer->setElementSize(sizeof(::ClippingPlane));
                this->clippingPlanesBuffer->setSize(0); // TODO bigger?

                this->launchParameters.clippingPlanesBuffer = this->clippingPlanesBuffer->getId();


                // Pick buffer
                this->pickBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER);
                this->pickBuffer->setElementSize(sizeof(PickStruct));
                this->pickBuffer->setSize(1);
                this->rayGenPickProgram["pickResult"]->setBuffer(this->pickBuffer);

                this->initialized = true;
            }

            /*
             * Programs
             */
            context->setRayGenerationProgram(RENDER_ENTRY_POINT, this->rayGenProgram);
            context->setMissProgram(RADIANCE_RAY_TYPE, this->missProgram);
            context->setRayGenerationProgram(BUFFER_CAST_ENTRY_POINT, this->bufferCastProgram);
            context->setRayGenerationProgram(PICK_ENTRY_POINT, this->rayGenPickProgram);
            context->setMissProgram(PICK_RAY_TYPE, this->missPickProgram);


            /*
             * Lights
             */
             // Update dirty lights
            for (VisRTX::Light* light : this->lights)
            {
                VisRTX::Impl::Light* l = dynamic_cast<VisRTX::Impl::Light*>(light);
                if (l->dirty)
                {
                    this->lightsNeedUpdate = true;
                    l->dirty = false;
                }
            }

            // Keep list of light geometries up to date
            VisRTX::Impl::Model* modelImpl = dynamic_cast<VisRTX::Impl::Model*>(this->model);
            modelImpl->UpdateLightGeometries(&this->lights);

            // Update light buffers
            if (this->lightsNeedUpdate)
            {
                this->lightsNeedUpdate = false;

                std::vector<::Light> activeDirectLights;
                std::vector<::Light> activeMissLights;

                this->pickLights.clear();

                for (VisRTX::Light* light : this->lights)
                {
                    ::Light tmp;
                    if (dynamic_cast<VisRTX::Impl::Light*>(light)->Set(tmp))
                    {
                        // Add light to light buffer
                        if (tmp.type == ::Light::POSITIONAL
                            || tmp.type == ::Light::QUAD
                            || tmp.type == ::Light::SPOT
                            || tmp.type == ::Light::DIRECTIONAL)
                        {
                            activeDirectLights.push_back(tmp);
                        }

                        if (tmp.type == ::Light::AMBIENT
                            || tmp.type == ::Light::HDRI
                            || (tmp.type == ::Light::DIRECTIONAL && tmp.angularDiameter > 0.0f))
                        {
                            activeMissLights.push_back(tmp);
                        }

                        // Update pick lights
                        this->pickLights[tmp.id] = light;
                    }
                }

                // Update buffers
                this->directLightsBuffer->setSize(activeDirectLights.size());
                if (!activeDirectLights.empty())
                {
                    memcpy(this->directLightsBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), &activeDirectLights[0], activeDirectLights.size() * sizeof(::Light));
                    this->directLightsBuffer->unmap();
                }

                this->missLightsBuffer->setSize(activeMissLights.size());
                if (!activeMissLights.empty())
                {
                    memcpy(this->missLightsBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), &activeMissLights[0], activeMissLights.size() * sizeof(::Light));
                    this->missLightsBuffer->unmap();
                }

                UPDATE_LAUNCH_PARAMETER((int)activeDirectLights.size(), this->launchParameters.numLightsDirect);
                UPDATE_LAUNCH_PARAMETER((int)activeMissLights.size(), this->launchParameters.numLightsMiss);                
            }

            // Check if any hit is required
            int disableAnyHit = (this->launchParameters.numClippingPlanes <= 0) ? 1 : 0;
            if (disableAnyHit)
            {
                for (VisRTX::Light* light : this->lights)
                {
                    VisRTX::Impl::Light* l = dynamic_cast<VisRTX::Impl::Light*>(light);
                    if (!l->visible)
                    {
                        disableAnyHit = 0;
                        break;
                    }

                    if (l->GetType() == VisRTX::LightType::QUAD)
                    {
                        VisRTX::Impl::QuadLight* q = dynamic_cast<VisRTX::Impl::QuadLight*>(light);
                        if (!q->twoSided)
                        {
                            disableAnyHit = 0;
                            break;
                        }
                    }
                }
            }

            UPDATE_LAUNCH_PARAMETER(disableAnyHit, this->launchParameters.disableAnyHit);

            return true;
        }

        bool Renderer::Pick(const Vec2f& screenPos, PickResult& result)
        {
            if (!this->Init())
                return false;

            optix::Context context = OptiXContext::Get();

            // Compute camera ray
            VisRTX::Impl::Camera* camera = dynamic_cast<VisRTX::Impl::Camera*>(this->camera);

            optix::float3 rayOrigin;
            optix::float3 rayDirection;

            const Vec2f pixel(2.0f * screenPos.x - 1.0f, 2.0f * (1.0f - screenPos.y) - 1.0f);

            const optix::float3 W = optix::normalize(camera->direction);
            const optix::float3 U = optix::normalize(optix::cross(camera->direction, camera->up));
            const optix::float3 V = optix::normalize(optix::cross(U, W));

            if (this->camera->GetType() == CameraType::PERSPECTIVE)
            {
                VisRTX::Impl::PerspectiveCamera* perspectiveCam = dynamic_cast<VisRTX::Impl::PerspectiveCamera*>(this->camera);

                const float vlen = tanf(0.5f * perspectiveCam->fovy * M_PIf / 180.0f);
                const float ulen = vlen * perspectiveCam->aspect;

                rayOrigin = perspectiveCam->position;
                rayDirection = optix::normalize(pixel.x * ulen * U + pixel.y * vlen * V + W);

            }
            else if (this->camera->GetType() == CameraType::ORTHOGRAPHIC)
            {
                VisRTX::Impl::OrthographicCamera* orthoCam = dynamic_cast<VisRTX::Impl::OrthographicCamera*>(this->camera);

                rayOrigin = orthoCam->position + 0.5f * (pixel.x * orthoCam->height * orthoCam->aspect * U + pixel.y * orthoCam->height * V);
                rayDirection = W;
            }

            this->rayGenPickProgram["rayOrigin"]->setFloat(rayOrigin);
            this->rayGenPickProgram["rayDirection"]->setFloat(rayDirection);

            context->launch(PICK_ENTRY_POINT, 1);


            // Extract result
            bool hit = false;
            result.geometry = nullptr;
            result.geometryHit = false;
            result.light = nullptr;
            result.lightHit = false;
            result.primitiveIndex = 0;
            result.position = Vec3f(0.0f, 0.0f, 0.0f);

            PickStruct* pick = static_cast<PickStruct*>(this->pickBuffer->map(0, RT_BUFFER_MAP_READ));

            if (pick->geometryId != 0)
            {
                result.geometry = dynamic_cast<VisRTX::Impl::Model*>(this->model)->pickGeometries[pick->geometryId];
                result.geometryHit = true;
                result.primitiveIndex = pick->primIndex;
                hit = true;
            }

            if (pick->lightId != 0)
            {
                result.lightHit = true;
                result.light = this->pickLights[pick->lightId];
                hit = true;
            }

            if (hit)
            {
                optix::float3 pos = rayOrigin + pick->t * rayDirection;
                result.position = Vec3f(pos.x, pos.y, pos.z);
            }

            this->pickBuffer->unmap();

            return hit;
        }

        void Renderer::SetCamera(VisRTX::Camera* camera)
        {
            if (camera != this->camera)
            {
                if (this->camera)
                    this->camera->Release();

                this->camera = camera;

                if (this->camera)
                {
                    this->camera->Retain();

                    // Mark this camera as dirty to enforce update
                    dynamic_cast<VisRTX::Impl::Camera*>(this->camera)->dirty = true;
                }
            }
        }

        void Renderer::SetModel(VisRTX::Model* model)
        {
            if (model != this->model)
            {
                // Remove lights from current model before losing it
                if (this->model)
                {
                    dynamic_cast<VisRTX::Impl::Model*>(this->model)->UpdateLightGeometries(nullptr);
                    this->model->Release();
                }

                this->model = model;

                if (this->model)
                {
                    this->model->Retain();

                    optix::Context context = OptiXContext::Get();
                    VisRTX::Impl::Model* m = dynamic_cast<VisRTX::Impl::Model*>(this->model);

                    this->rayGenProgram["topObject"]->set(m->superGroup);
                    this->rayGenPickProgram["topObject"]->set(m->superGroup);
                }
            }
        }

        void Renderer::AddLight(VisRTX::Light* light)
        {
            if (!light)
                return;

            auto it = this->lights.find(light);
            if (it == this->lights.end())
            {
                this->lights.insert(light);
                light->Retain();
            }
        }

        void Renderer::RemoveLight(VisRTX::Light* light)
        {
            if (!light)
                return;

            auto it = this->lights.find(light);
            if (it != this->lights.end())
            {
                this->lights.erase(it);
                light->Release();
            }
        }

        void Renderer::SetClippingPlanes(uint32_t numPlanes, ClippingPlane* planes)
        {
            UPDATE_LAUNCH_PARAMETER(numPlanes, this->launchParameters.numClippingPlanes);

            if (numPlanes > 0)
            {
                this->clippingPlanesBuffer->setSize(numPlanes);

                ::ClippingPlane* dst = static_cast<::ClippingPlane*>(this->clippingPlanesBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

                for (uint32_t i = 0; i < numPlanes; ++i)
                {
                    dst[i].coefficients.x = planes[i].normal.x; // a
                    dst[i].coefficients.y = planes[i].normal.y; // b
                    dst[i].coefficients.z = planes[i].normal.z; // c                    
                    dst[i].coefficients.w = -optix::dot(make_float3(planes[i].position), make_float3(planes[i].normal)); // d
                    
                    dst[i].primaryRaysOnly = planes[i].primaryRaysOnly ? 1 : 0;
                }

                this->clippingPlanesBuffer->unmap();
            }          
        }

        void Renderer::SetToneMapping(bool enabled, float gamma, const Vec3f& colorBalance, float whitePoint, float burnHighlights, float crushBlacks, float saturation, float brightness)
        {
            if (this->toneMappingFixed && !this->ignoreOverrides)
                return;

            const int toneMapping = enabled ? 1 : 0;

            UPDATE_LAUNCH_PARAMETER(toneMapping, this->launchParameters.toneMapping);

            if (toneMapping)
            {
                const optix::float3 colorBalance2 = optix::make_float3(colorBalance.x, colorBalance.y, colorBalance.z);
                const float invGamma = 1.0f / gamma;
                const float invWhitePoint = brightness / whitePoint;
                const float crushBlacks2 = crushBlacks + crushBlacks + 1.0f;

                UPDATE_LAUNCH_PARAMETER(colorBalance2, this->launchParameters.colorBalance);
                UPDATE_LAUNCH_PARAMETER(invGamma, this->launchParameters.invGamma);
                UPDATE_LAUNCH_PARAMETER(invWhitePoint, this->launchParameters.invWhitePoint);
                UPDATE_LAUNCH_PARAMETER(burnHighlights, this->launchParameters.burnHighlights);
                UPDATE_LAUNCH_PARAMETER(crushBlacks2, this->launchParameters.crushBlacks);
                UPDATE_LAUNCH_PARAMETER(saturation, this->launchParameters.saturation);
            }
        }

        void Renderer::SetDenoiser(DenoiserType denoiser)
        {
            if (this->denoiserFixed && !this->ignoreOverrides)
                return;

            this->denoiser = denoiser;

            // TODO expose in API?
            this->blendDelay = 20;
            this->blendDuration = 1000;
        }

        void Renderer::SetSamplesPerPixel(uint32_t spp)
        {
            this->samplesPerPixel = spp;
        }

        void Renderer::SetEpsilon(float epsilon)
        {
            if (this->epsilonFixed && !this->ignoreOverrides)
                return;

            // Conservative epsilon...
            const float eps = std::max(1e-3f, epsilon);            
            UPDATE_LAUNCH_PARAMETER(eps, this->launchParameters.occlusionEpsilon);
        }

        void Renderer::SetAlphaCutoff(float alphaCutoff)
        {
            UPDATE_LAUNCH_PARAMETER(alphaCutoff, this->launchParameters.alphaCutoff);
        }

        void Renderer::SetNumBounces(uint32_t minBounces, uint32_t maxBounces)
        {
            if (!this->minBouncesFixed || this->ignoreOverrides)
                UPDATE_LAUNCH_PARAMETER((int) minBounces, this->launchParameters.numBouncesMin);

            if (!this->maxBouncesFixed || this->ignoreOverrides)
                UPDATE_LAUNCH_PARAMETER((int) maxBounces, this->launchParameters.numBouncesMax);
        }

        void Renderer::SetWriteBackground(bool writeBackground)
        {
            const int wb = writeBackground ? 1 : 0;
            UPDATE_LAUNCH_PARAMETER(wb, this->launchParameters.writeBackground);
        }

        void Renderer::SetFireflyClamping(float direct, float indirect)
        {
            if (!this->clampDirectFixed || this->ignoreOverrides)
                UPDATE_LAUNCH_PARAMETER(direct, this->launchParameters.fireflyClampingDirect);

            if (!this->clampIndirectFixed || this->ignoreOverrides)
                UPDATE_LAUNCH_PARAMETER(indirect, this->launchParameters.fireflyClampingIndirect);
        }

        void Renderer::SetSampleAllLights(bool sampleAllLights)
        {
            if (!this->sampleAllLightsFixed || this->ignoreOverrides)
            {
                const int all = sampleAllLights ? 1 : 0;
                UPDATE_LAUNCH_PARAMETER(all, this->launchParameters.sampleAllLights);
            }
        }
    }
}
