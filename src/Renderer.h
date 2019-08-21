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
#include "Geometry.h"
#include "Object.h"

#include "Pathtracer/Common.h"

#include <set>
#include <map>

namespace VisRTX
{
    namespace Impl
    {
        class FrameBuffer;

        class Renderer : public virtual Object, public VisRTX::Renderer
        {
        public:
            Renderer();
            ~Renderer();

        public:
            void Render(VisRTX::FrameBuffer* frameBuffer) override;
            bool Pick(const Vec2f& screenPos, PickResult& result) override;

            void SetCamera(VisRTX::Camera* camera) override;
            void SetModel(VisRTX::Model* model) override;

            void AddLight(VisRTX::Light* light) override;
            void RemoveLight(VisRTX::Light* light) override;

            void SetClippingPlanes(uint32_t numPlanes, ClippingPlane* planes) override;

            void SetToneMapping(bool enabled, float gamma, const Vec3f& colorBalance, float whitePoint, float burnHighlights, float crushBlacks, float saturation, float brightness) override;

            void SetDenoiser(DenoiserType denoiser) override;

            void SetSamplesPerPixel(uint32_t spp) override;
            void SetAlphaCutoff(float alphaCutoff) override;
            void SetNumBounces(uint32_t minBounces, uint32_t maxBounces) override;
            void SetWriteBackground(bool writeBackground) override;
            void SetFireflyClamping(float direct, float indirect) override;
            void SetSampleAllLights(bool sampleAllLights) override;
            void SetEpsilon(float epsilon) override;

        private:
            bool Init();            

        private:
            bool initialized = false;
            bool contextValidated = false;

            optix::Program rayGenProgram;
            optix::Program rayGenPickProgram;
            optix::Program missProgram;                        
            optix::Program missPickProgram;
            optix::Program bufferCastProgram;

            VisRTX::Camera* camera = nullptr;
            VisRTX::Model* model = nullptr;

            DenoiserType denoiser;
            uint32_t blendDelay;
            uint32_t blendDuration;

            uint32_t samplesPerPixel;

            optix::Buffer basicMaterialParametersBuffer;
            optix::Buffer mdlMaterialParametersBuffer;

            bool lightsNeedUpdate = true;
            std::set<VisRTX::Light*> lights;
            optix::Buffer directLightsBuffer; // lights used for next event estimation
            optix::Buffer missLightsBuffer; // lights used in miss program           

            optix::Buffer clippingPlanesBuffer;

            optix::Buffer pickBuffer;
            std::map<int, VisRTX::Light*> pickLights;

            // Env var overrides
            bool ignoreOverrides = false;
            bool minBouncesFixed = false;
            bool maxBouncesFixed = false;
            bool toneMappingFixed = false;
            bool denoiserFixed = false;
            bool clampDirectFixed = false;
            bool clampIndirectFixed = false;
            bool sampleAllLightsFixed = false;
            
            LaunchParameters launchParameters;
            bool launchParametersDirty = true;
            optix::Buffer launchParametersBuffer;
        };
    }
}

#pragma warning( push )