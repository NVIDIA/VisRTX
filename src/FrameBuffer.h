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

#include <map>

namespace VisRTX
{
    namespace Impl
    {        
        class FrameBuffer : public virtual Object, public VisRTX::FrameBuffer
        {
            friend class Renderer;

        public:
            FrameBuffer(FrameBufferFormat format, const Vec2ui& size = Vec2ui(1920, 1080));
            ~FrameBuffer();

        public:
            void Resize(const Vec2ui& size) override;

            void Clear() override;

            FrameBufferFormat GetFormat() override;

            const void* MapColorBuffer() override;
            const float* MapDepthBuffer() override;
            void Unmap(const void* mapped) override;

            uint32_t GetColorTextureGL() override;
            uint32_t GetDepthTextureGL() override;

            void SetDepthNormalization(float clipMin, float clipMax) override;

        private:
            void UpdateTexture(uint32_t& texture, cudaGraphicsResource_t& resource, Vec2ui& size, GLint internalFormat, GLenum format, GLenum type, optix::Buffer src, uint32_t bytesPerComponent);

        private:
            uint32_t width = 0;
            uint32_t height = 0;

            FrameBufferFormat format;

            uint32_t frameNumber = 0;

            float depthClipMin = -1.0f;
            float depthClipMax = -1.0f;
            float depthClipDiv = 1.0f;

            optix::Buffer accumulationBuffer;
            optix::Buffer frameBuffer;
            optix::Buffer ucharFrameBuffer;
            optix::Buffer depthBuffer;
           
            optix::PostprocessingStage denoiserStage;
            optix::Variable denoiserBlend;
            optix::CommandList denoiserCommandList;                        

            std::map<const void*, optix::Buffer> mappedBuffers;
            
            uint32_t colorTexture = 0;
            cudaGraphicsResource_t colorTextureGraphicsResource = nullptr;
            Vec2ui colorTextureSize;

            uint32_t depthTexture = 0;
            cudaGraphicsResource_t depthTextureGraphicsResource = nullptr;
            Vec2ui depthTextureSize;

            bool forceMapGL = false;
        };
    }
}

#pragma warning( pop )