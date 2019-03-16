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

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif


#include "Config.h"

#include <GL/glew.h>

#include <cuda_gl_interop.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#define OPTIX_VERSION_MAJOR (OPTIX_VERSION / 10000)
#define OPTIX_VERSION_MINOR ((OPTIX_VERSION % 10000) / 100)
#define OPTIX_VERSION_MICRO (OPTIX_VERSION % 100)

#include "VisRTX.h"

#include <iostream>
#include <iomanip>


#define RENDER_ENTRY_POINT          0
#define BUFFER_CAST_ENTRY_POINT     1
#define PICK_ENTRY_POINT            2


inline void usageReportCallback(int lvl, const char* tag, const char* msg, void* /*cbdata*/)
{
    std::cout << "[" << lvl << "][" << std::left << std::setw(12) << tag << "] " << msg;
}

namespace VisRTX
{
    inline bool Destroy(optix::DestroyableObj* obj)
    {
        if (obj)
        {
            obj->destroy();
            return true;
        }
        return false;
    }

    inline optix::float3 make_float3(const Vec3f& v)
    {
        return optix::make_float3(v.x, v.y, v.z);
    }

    inline optix::float4 make_float4(const Vec4f& v)
    {
        return optix::make_float4(v.x, v.y, v.z, v.w);
    }

    inline optix::uint3 make_uint3(const Vec3ui& v)
    {
        return optix::make_uint3(v.x, v.y, v.z);
    }

    /*
     * OptiX context as a singleton
     */
    class OptiXContext
    {
    public:
        static optix::Context Get()
        {
            static bool initialized = false;
            static optix::Context context;

            if (!initialized)
            {
                try
                {
                    context = optix::Context::create();
                    Init(context);
                }
                catch (optix::Exception& e)
                {
                    throw VisRTX::Exception(VisRTX::Error::UNKNOWN_ERROR, e.getErrorString().c_str());
                }
                
                initialized = true;
            }

            return context;
        }

    private:
        static void Init(optix::Context& context);
    };


    /**
     * Creates and returns singleton dummy objects which are required for initializing declared but unused variables.
     */
    class Dummy
    {
    public:
        static optix::Buffer GetBuffer()
        {
            static optix::Buffer buffer = OptiXContext::Get()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, 1u, 1u);
            unsigned char* buffer_data = static_cast<unsigned char*>(buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
            buffer_data[0] = 255;
            buffer_data[1] = 255;
            buffer_data[2] = 255;
            buffer_data[3] = 255;
            buffer->unmap();

            return buffer;
        }

        static optix::TextureSampler GetTextureSampler()
        {
            static optix::TextureSampler sampler = OptiXContext::Get()->createTextureSampler();
            sampler->setWrapMode(0, RT_WRAP_REPEAT);
            sampler->setWrapMode(1, RT_WRAP_REPEAT);
            sampler->setWrapMode(2, RT_WRAP_REPEAT);
            sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
            sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
            sampler->setMaxAnisotropy(1.0f);
            sampler->setMipLevelCount(1u);
            sampler->setArraySize(1u);
            sampler->setBuffer(0u, 0u, Dummy::GetBuffer());
            sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

            return sampler;
        }
    };
}

