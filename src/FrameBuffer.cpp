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

#include "FrameBuffer.h"
#include <cuda_runtime_api.h>

namespace VisRTX
{
    namespace Impl
    {
        inline void CUDA_THROW(cudaError_t code, const std::string& errorMessage)
        {
            if (cudaSuccess != code) 
            {
                const std::string str = errorMessage + " (Error " + std::to_string(code) + ": " + std::string(cudaGetErrorString(code)) + ")";
                throw VisRTX::Exception(VisRTX::Error::UNKNOWN_ERROR, str.c_str());
            }
        }

        FrameBuffer::FrameBuffer(FrameBufferFormat format, const Vec2ui& size)
        {
            // Env var overrides
            if (const char* str = std::getenv("VISRTX_GL_FORCE_MAP"))
            {
                this->forceMapGL = atoi(str) > 0;
            }

            this->format = format;

            optix::Context context = OptiXContext::Get();

            // Buffers
            this->accumulationBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4);
            this->frameBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4);
            this->ucharFrameBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4);
            this->depthBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT);

            // AI denoiser postprocessing
            try
            {
                this->denoiserStage = context->createBuiltinPostProcessingStage("DLDenoiser");
                this->denoiserStage->declareVariable("input_buffer")->set(this->frameBuffer);
                this->denoiserStage->declareVariable("output_buffer")->set(this->frameBuffer);
                this->denoiserBlend = this->denoiserStage->declareVariable("blend");
                this->denoiserBlend->setFloat(0.0f);
            }
            catch (optix::Exception e)
            {
                // std::cout << "Warning: Could not find OptiX DL denoiser. Are you using an OptiX version that does not support it?" << std::endl;
                this->denoiserStage = 0;
            }

            this->Resize(size);

            this->SetDepthNormalization(-1.0f, -1.0f); // no normalization
        }

        FrameBuffer::~FrameBuffer()
        {
            if (this->colorTextureGraphicsResource)
                cudaGraphicsUnregisterResource(this->colorTextureGraphicsResource);

            if (this->colorTexture)
                glDeleteTextures(1, &this->colorTexture);

            if (this->depthTextureGraphicsResource)
                cudaGraphicsUnregisterResource(this->depthTextureGraphicsResource);

            if (this->depthTexture)
                glDeleteTextures(1, &this->depthTexture);

            Destroy(this->accumulationBuffer.get());
            Destroy(this->frameBuffer.get());
            Destroy(this->ucharFrameBuffer.get());
            Destroy(this->depthBuffer.get());
            Destroy(this->denoiserStage.get());
            Destroy(this->denoiserCommandList.get());
        }

        void FrameBuffer::Resize(const Vec2ui& size)
        {
            const uint32_t width = size.width;
            const uint32_t height = size.height;

            if (width == this->width && height == this->height)
                return;

            this->width = width;
            this->height = height;

            // Resize buffers
            this->accumulationBuffer->setSize(width, height);
            this->frameBuffer->setSize(width, height);
            this->ucharFrameBuffer->setSize(width, height);
            this->depthBuffer->setSize(width, height);

            // Recreate AI denoiser command list
            if (this->denoiserStage)
            {
                if (this->denoiserCommandList)
                    this->denoiserCommandList->destroy();

                this->denoiserCommandList = OptiXContext::Get()->createCommandList();
                this->denoiserCommandList->appendLaunch(RENDER_ENTRY_POINT, width, height);
                this->denoiserCommandList->appendPostprocessingStage(this->denoiserStage, width, height);
                
                if (this->format != FrameBufferFormat::RGBA32F)
                    this->denoiserCommandList->appendLaunch(BUFFER_CAST_ENTRY_POINT, width, height);

                this->denoiserCommandList->finalize();
            }

            // Clear
            this->Clear();
        }

        void FrameBuffer::Clear()
        {
            // We don't actually clear the contents, just reset the accumulation counter for now.
            this->frameNumber = 0;
        }

        FrameBufferFormat FrameBuffer::GetFormat()
        {
            return this->format;
        }

        const void* FrameBuffer::MapColorBuffer()
        {
            optix::Buffer mappedBuffer = this->ucharFrameBuffer;

            if (this->format == FrameBufferFormat::RGBA32F)
                mappedBuffer = this->frameBuffer;


            const void* mapped = mappedBuffer->map(0, RT_BUFFER_MAP_READ);
            this->mappedBuffers[mapped] = mappedBuffer;
            return mapped;
        }

        const float* FrameBuffer::MapDepthBuffer()
        {
            const void* mapped = this->depthBuffer->map(0, RT_BUFFER_MAP_READ);
            this->mappedBuffers[mapped] = this->depthBuffer;
            return reinterpret_cast<const float*>(mapped);
        }

        void FrameBuffer::Unmap(const void* mapped)
        {
            auto it = this->mappedBuffers.find(mapped);

            if (it != this->mappedBuffers.end())
            {
                it->second->unmap();
                this->mappedBuffers.erase(it);
            }
        }

        void FrameBuffer::SetDepthNormalization(float clipMin, float clipMax)
        {
            this->depthClipMin = clipMin;
            this->depthClipMax = clipMax;

            const float delta = clipMax - clipMin;
            this->depthClipDiv = (delta != 0.0f) ? 1.0f / delta : 1.0f;
        }

        uint32_t FrameBuffer::GetColorTextureGL()
        {
            if (this->format == FrameBufferFormat::RGBA8)
                this->UpdateTexture(this->colorTexture, this->colorTextureGraphicsResource, this->colorTextureSize, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, this->ucharFrameBuffer, 4);            
            else if (this->format == FrameBufferFormat::RGBA32F)
                this->UpdateTexture(this->colorTexture, this->colorTextureGraphicsResource, this->colorTextureSize, GL_RGBA32F, GL_RGBA, GL_FLOAT, this->frameBuffer, 16);

            return this->colorTexture;
        }

        uint32_t FrameBuffer::GetDepthTextureGL()
        {
            this->UpdateTexture(this->depthTexture, this->depthTextureGraphicsResource, this->depthTextureSize, GL_R32F, GL_RED, GL_FLOAT, this->depthBuffer, 4);

            return this->depthTexture;
        }

        void FrameBuffer::UpdateTexture(uint32_t& texture, cudaGraphicsResource_t& resource, Vec2ui& size, GLint internalFormat, GLenum format, GLenum type, optix::Buffer src, uint32_t bytesPerComponent)
        {
            // Make sure OpenGL is initialized
            if (!OpenGL::Init())
            {
                texture = 0;
                return;    
            }

            // Init
            if (!texture)
                glGenTextures(1, &texture);

            if (this->forceMapGL)
            {
                const void* pixels = src->map(0, RT_BUFFER_MAP_READ);

                glBindTexture(GL_TEXTURE_2D, texture);
                glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, this->width, this->height, 0, format, type, pixels);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

                src->unmap();

                size = Vec2ui(width, height);
            }
            else
            {
                // Resize
                if (size.width != this->width || size.height != this->height)
                {
                    if (resource)
                    {
                        CUDA_THROW(cudaGraphicsUnregisterResource(resource),
                            "Failed to unregister graphics resource.");
                        resource = nullptr;
                    }

                    glBindTexture(GL_TEXTURE_2D, texture);
                    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, this->width, this->height, 0, format, type, 0);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

                    CUDA_THROW(cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard),
                        "Failed to register image as graphics resource.");

                    size = Vec2ui(width, height);
                }

                const std::vector<int> enabledDevices = OptiXContext::Get()->getEnabledDevices();
                const int interopDevice = enabledDevices.size() > 0 ? enabledDevices[0] : 0; // TODO actually determine the device which drives the OpenGL context

                // Update
                CUDA_THROW(cudaGraphicsMapResources(1, &resource),
                    "Failed to map graphics resource.");
                cudaArray_t array;
                CUDA_THROW(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0),
                    "Failed to get mapped array of graphics resource.");
                CUDA_THROW(cudaMemcpy2DToArray(array, 0, 0, src->getDevicePointer(interopDevice), this->width * bytesPerComponent, this->width * bytesPerComponent, this->height, cudaMemcpyDeviceToDevice),
                    "Failed to copy to graphics resource array.");
                CUDA_THROW(cudaGraphicsUnmapResources(1, &resource),
                    "Failed to unmap graphics resource.");
            }
        }
    }
}
