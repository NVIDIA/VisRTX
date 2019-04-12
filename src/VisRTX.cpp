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

#include "OptiX.h"

#include "VisRTX.h"

#include "Renderer.h"
#include "FrameBuffer.h"
#include "Camera.h"
#include "Light.h"
#include "Geometry.h"
#include "Texture.h"
#include "Material.h"
#include "Model.h"

#include <array>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <string>

namespace VisRTX
{
    namespace Impl
    {
        class Context : public VisRTX::Context
        {
        private:
            struct Device
            {
                std::string name;
                uint64_t memoryTotal;
            };

            std::vector<Device> devices;

        public:
            Context()
            {
                // Query all available devices
                unsigned int count;
                if (RT_SUCCESS == rtDeviceGetDeviceCount(&count))
                {
                    for (unsigned int i = 0; i < count; ++i)
                    {
                        const uint32_t maxName = 1024;
                        char name[maxName];
                        memset(name, 0, maxName);

                        rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, maxName, name);

                        RTsize size = 0;
                        rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(RTsize), &size);

                        Device device;
                        device.name = std::string(name);
                        device.memoryTotal = size;

                        this->devices.push_back(device);
                    }
                }
            }

            uint32_t GetDeviceCount()
            {
                return (uint32_t)this->devices.size();
            }

            const char* GetDeviceName(uint32_t device)
            {
                return this->devices[device].name.c_str();
            }

            uint64_t GetDeviceMemoryTotal(uint32_t device)
            {
                return this->devices[device].memoryTotal;
            }

            uint64_t GetDeviceMemoryAvailable(uint32_t device)
            {
                try
                {
                    return OptiXContext::Get()->getAvailableDeviceMemory(device);
                }
                catch (optix::Exception e)
                {
                    return 0;
                }
            }

            bool SetDevices(uint32_t numDevices, uint32_t* devices)
            {
                try
                {
                    std::vector<int> devicesInt;
                    for (uint32_t i = 0; i < numDevices; ++i)
                        devicesInt.push_back(static_cast<int>(devices[i]));

                    OptiXContext::Get()->setDevices(std::begin(devicesInt), std::end(devicesInt));
                    return true;
                }
                catch (optix::Exception e)
                {
                    return false;
                }
            }

        public:
            VisRTX::Renderer* CreateRenderer()
            {
                return new VisRTX::Impl::Renderer();
            }

            VisRTX::FrameBuffer* CreateFrameBuffer(FrameBufferFormat format)
            {
                return new VisRTX::Impl::FrameBuffer(format);
            }

            VisRTX::FrameBuffer* CreateFrameBuffer(FrameBufferFormat format, const Vec2ui& size)
            {
                return new VisRTX::Impl::FrameBuffer(format, size);
            }

            VisRTX::PerspectiveCamera* CreatePerspectiveCamera()
            {
                return new VisRTX::Impl::PerspectiveCamera();
            }

            VisRTX::PerspectiveCamera* CreatePerspectiveCamera(const Vec3f& position, const Vec3f& direction, const Vec3f& up, float fovy)
            {
                return new VisRTX::Impl::PerspectiveCamera(position, direction, up, fovy);
            }

            VisRTX::OrthographicCamera* CreateOrthographicCamera()
            {
                return new VisRTX::Impl::OrthographicCamera();
            }

            VisRTX::OrthographicCamera* CreateOrthographicCamera(const Vec3f& position, const Vec3f& direction, const Vec3f& up, float height)
            {
                return new VisRTX::Impl::OrthographicCamera(position, direction, up, height);
            }

            VisRTX::SphericalLight* CreateSphericalLight()
            {
                return new VisRTX::Impl::SphericalLight();
            }

            VisRTX::SphericalLight* CreateSphericalLight(const Vec3f& position, const Vec3f& color, float radius)
            {
                return new VisRTX::Impl::SphericalLight(position, color, radius);
            }

            VisRTX::DirectionalLight* CreateDirectionalLight()
            {
                return new VisRTX::Impl::DirectionalLight();
            }

            VisRTX::DirectionalLight* CreateDirectionalLight(const Vec3f& direction, const Vec3f& color)
            {
                return new VisRTX::Impl::DirectionalLight(direction, color);
            }

            VisRTX::QuadLight* CreateQuadLight()
            {
                return new VisRTX::Impl::QuadLight();
            }

            VisRTX::QuadLight* CreateQuadLight(const Vec3f& position, const Vec3f& edge1, const Vec3f& edge2, const Vec3f& color)
            {
                return new VisRTX::Impl::QuadLight(position, edge1, edge2, color);
            }

            VisRTX::SpotLight* CreateSpotLight()
            {
                return new VisRTX::Impl::SpotLight();
            }

            VisRTX::SpotLight* CreateSpotLight(const Vec3f& position, const Vec3f& direction, const Vec3f& color, float openingAngle, float penumbraAngle, float radius)
            {
                return new VisRTX::Impl::SpotLight(position, direction, color, openingAngle, penumbraAngle, radius);
            }

            VisRTX::AmbientLight* CreateAmbientLight()
            {
                return new VisRTX::Impl::AmbientLight();
            }

            VisRTX::AmbientLight* CreateAmbientLight(const Vec3f& color)
            {
                return new VisRTX::Impl::AmbientLight(color);
            }

            VisRTX::HDRILight* CreateHDRILight()
            {
                return new VisRTX::Impl::HDRILight();
            }

            VisRTX::HDRILight* CreateHDRILight(VisRTX::Texture* texture)
            {
                return new VisRTX::Impl::HDRILight(texture);
            }

            VisRTX::TriangleGeometry* CreateTriangleGeometry()
            {
                return new VisRTX::Impl::TriangleGeometry();
            }

            VisRTX::TriangleGeometry* CreateTriangleGeometry(uint32_t numTriangles, const Vec3ui* triangles, uint32_t numVertices, const Vec3f* vertices, const Vec3f* normals)
            {
                return new VisRTX::Impl::TriangleGeometry(numTriangles, triangles, numVertices, vertices, normals);
            }

            VisRTX::SphereGeometry* CreateSphereGeometry()
            {
                return new VisRTX::Impl::SphereGeometry();
            }

            VisRTX::SphereGeometry* CreateSphereGeometry(uint32_t numVertices, const Vec3f* vertices, const float* radii)
            {
                return new VisRTX::Impl::SphereGeometry(numVertices, vertices, radii);
            }

            VisRTX::CylinderGeometry* CreateCylinderGeometry()
            {
                return new VisRTX::Impl::CylinderGeometry();
            }

            VisRTX::CylinderGeometry* CreateCylinderGeometry(uint32_t numCylinders, const Vec2ui* cylinders, uint32_t numVertices, const Vec3f* vertices, const float* radii)
            {
                return new VisRTX::Impl::CylinderGeometry(numCylinders, cylinders, numVertices, vertices, radii);
            }

            VisRTX::DiskGeometry* CreateDiskGeometry()
            {
                return new VisRTX::Impl::DiskGeometry();
            }

            VisRTX::DiskGeometry* CreateDiskGeometry(uint32_t numVertices, const Vec3f* vertices, const Vec3f* normals, const float* radii)
            {
                return new VisRTX::Impl::DiskGeometry(numVertices, vertices, normals, radii);
            }

            VisRTX::Texture* CreateTexture()
            {
                return new VisRTX::Impl::Texture();
            }

            VisRTX::Texture* CreateTexture(const Vec2ui& size, TextureFormat format, const void* src)
            {
                return new VisRTX::Impl::Texture(size, format, src);
            }

            VisRTX::BasicMaterial* CreateBasicMaterial()
            {
                return new VisRTX::Impl::BasicMaterial();
            }

            VisRTX::MDLMaterial* CreateMDLMaterial()
            {
                return new VisRTX::Impl::MDLMaterial();
            }

            VisRTX::MDLMaterial* CreateMDLMaterial(const char* material, const char* source, uint32_t sourceBytes, uint32_t numModulePaths, const char** modulePaths, CompilationType compilationType, uint8_t priority)
            {
                return new VisRTX::Impl::MDLMaterial(material, source, sourceBytes, numModulePaths, modulePaths, compilationType, priority);
            }

            VisRTX::Model* CreateModel()
            {
                return new VisRTX::Impl::Model();
            }
        };
    }
}



VisRTX::Context* VisRTX_GetContext()
{
    static VisRTX::Impl::Context context;

    static bool devicesSelected = false;
    if (!devicesSelected && context.GetDeviceCount() > 0)
    {
        std::vector<uint32_t> devices;

        if (const char* devicesString = std::getenv("VISRTX_DEVICES"))
        {
            const std::string str = std::string(devicesString);
            std::stringstream ss(str);
            uint32_t i;

            while (ss >> i)
            {
                if (i < context.GetDeviceCount())
                    devices.push_back(i);

                if (ss.peek() == ',')
                    ss.ignore();
            }
        }

        if (devices.empty())
        {
            // Use all available devices by default
            uint32_t n = context.GetDeviceCount();
            for (uint32_t i = 0; i < n; ++i)
                devices.push_back(i);
        }

        context.SetDevices((uint32_t)devices.size(), devices.data());

        std::cout << "VisRTX " << VISRTX_VERSION << ", using devices:" << std::endl;
        for (uint32_t i : devices)
        {
            std::string name = context.GetDeviceName(i);
            uint64_t totalMem = context.GetDeviceMemoryTotal(i);
            uint64_t availMem = context.GetDeviceMemoryAvailable(i);

            float totalGB = totalMem * 1e-9f;
            float availGB = availMem * 1e-9f;

            std::cout << std::fixed << std::setprecision(1) << " " << i << ": " << name << " (Total: " << totalGB << " GB, Available: " << availGB << " GB)" << std::endl;
        }

        devicesSelected = true;
    }

    return &context;
}
