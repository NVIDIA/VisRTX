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

#include "Texture.h"

#include <cstring>

namespace VisRTX
{
    namespace Impl
    {
        RTfiltermode convertFilterMode(TextureFiltering f)
        {
            if (f == TextureFiltering::NEAREST)
                return RT_FILTER_NEAREST;
            return RT_FILTER_LINEAR;
        }

        RTwrapmode convertWrapMode(TextureWrapMode w)
        {
            if (w == TextureWrapMode::CLAMP_TO_EDGE)
                return RT_WRAP_CLAMP_TO_EDGE;
            if (w == TextureWrapMode::MIRROR)
                return RT_WRAP_MIRROR;
            return RT_WRAP_REPEAT;
        }

        Texture::Texture()
        {
            optix::Context context = OptiXContext::Get();

            this->sampler = context->createTextureSampler();
            this->sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
            this->sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
            this->sampler->setMipLevelCount(1u);
            this->sampler->setArraySize(1u);

            this->SetFiltering(TextureFiltering::LINEAR, TextureFiltering::LINEAR);
            this->SetWrapMode(TextureWrapMode::REPEAT, TextureWrapMode::REPEAT);
            this->SetMaxAnisotropy(1.0f);
        }

        Texture::Texture(const Vec2ui& size, TextureFormat format, const void* src) : Texture()
        {
            this->SetPixels(size, format, src);
        }

        Texture::~Texture()
        {
            Destroy(this->sampler.get());
            Destroy(this->buffer.get());
        }

        void Texture::SetPixels(const Vec2ui& size, TextureFormat format, const void* src)
        {
            uint32_t width = size.width;
            uint32_t height = size.height;

            optix::Context context = OptiXContext::Get();

            if (!this->buffer)
            {
                this->buffer = context->createBuffer(RT_BUFFER_INPUT);
                this->sampler->setBuffer(0u, 0u, this->buffer);
            }

            if (format == TextureFormat::RGBA8)
            {
                this->buffer->setFormat(RT_FORMAT_UNSIGNED_BYTE4);
                this->buffer->setSize(width, height);

                memcpy(this->buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), src, width * height * sizeof(optix::uchar4));
                this->buffer->unmap();

                this->sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
            }

            else if (format == TextureFormat::RGB8)
            {
                this->buffer->setFormat(RT_FORMAT_UNSIGNED_BYTE4);
                this->buffer->setSize(width, height);

                optix::uchar4* dst = static_cast<optix::uchar4*>(this->buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

                for (uint32_t y = 0; y < height; ++y)
                {
                    for (uint32_t x = 0; x < width; ++x)
                    {
                        optix::uchar4* dstPixel = dst + y * width + x;
                        const optix::uchar3* srcPixel = static_cast<const optix::uchar3*>(src) + y * width + x;

                        dstPixel->x = srcPixel->x;
                        dstPixel->y = srcPixel->y;
                        dstPixel->z = srcPixel->z;
                        dstPixel->w = 255;
                    }
                }

                this->buffer->unmap();

                this->sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
            }

            else if (format == TextureFormat::R8)
            {
                this->buffer->setFormat(RT_FORMAT_UNSIGNED_BYTE4);
                this->buffer->setSize(width, height);

                optix::uchar4* dst = static_cast<optix::uchar4*>(this->buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

                for (uint32_t y = 0; y < height; ++y)
                {
                    for (uint32_t x = 0; x < width; ++x)
                    {
                        optix::uchar4* dstPixel = dst + y * width + x;
                        const uint8_t* srcPixel = static_cast<const uint8_t*>(src) + y * width + x;

                        dstPixel->x = *srcPixel;
                        dstPixel->y = *srcPixel;
                        dstPixel->z = *srcPixel;
                        dstPixel->w = 255;
                    }
                }

                this->buffer->unmap();

                this->sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
            }

            else if (format == TextureFormat::RGBA32F)
            {
                this->buffer->setFormat(RT_FORMAT_FLOAT4);
                this->buffer->setSize(width, height);

                memcpy(this->buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), src, width * height * sizeof(optix::float4));
                this->buffer->unmap();

                this->sampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
            }

            else if (format == TextureFormat::RGB32F)
            {
                this->buffer->setFormat(RT_FORMAT_FLOAT4);
                this->buffer->setSize(width, height);

                optix::float4* dst = static_cast<optix::float4*>(this->buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

                for (uint32_t y = 0; y < height; ++y)
                {
                    for (uint32_t x = 0; x < width; ++x)
                    {
                        optix::float4* dstPixel = dst + y * width + x;
                        const optix::float3* srcPixel = static_cast<const optix::float3*>(src) + y * width + x;

                        dstPixel->x = srcPixel->x;
                        dstPixel->y = srcPixel->y;
                        dstPixel->z = srcPixel->z;
                        dstPixel->w = 1.0f;
                    }
                }

                this->buffer->unmap();

                this->sampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
            }

            else if (format == TextureFormat::R32F)
            {
                this->buffer->setFormat(RT_FORMAT_FLOAT4);
                this->buffer->setSize(width, height);

                optix::float4* dst = static_cast<optix::float4*>(this->buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

                for (uint32_t y = 0; y < height; ++y)
                {
                    for (uint32_t x = 0; x < width; ++x)
                    {
                        optix::float4* dstPixel = dst + y * width + x;
                        const float* srcPixel = static_cast<const float*>(src) + y * width + x;

                        dstPixel->x = *srcPixel;
                        dstPixel->y = *srcPixel;
                        dstPixel->z = *srcPixel;
                        dstPixel->w = 1.0f;
                    }
                }

                this->buffer->unmap();

                this->sampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
            }

            else
            {
                throw Exception(Error::INVALID_ARGUMENT, "Invalid texture format.");
            }
        }

        void Texture::SetFiltering(TextureFiltering minification, TextureFiltering magnification)
        {
            this->sampler->setFilteringModes(convertFilterMode(minification), convertFilterMode(magnification), RT_FILTER_NONE);
        }

        void Texture::SetWrapMode(TextureWrapMode wrapModeU, TextureWrapMode wrapModeV)
        {
            this->sampler->setWrapMode(0, convertWrapMode(wrapModeU));
            this->sampler->setWrapMode(1, convertWrapMode(wrapModeV));
        }

        void Texture::SetMaxAnisotropy(float maxAnisotropy)
        {
            this->sampler->setMaxAnisotropy(maxAnisotropy);
        }
    }
}
