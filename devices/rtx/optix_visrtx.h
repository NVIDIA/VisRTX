/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "gpu/gpu_objects.h"
#include "utility/DeferredArrayUploadBuffer.h"
#include "utility/DeviceObjectArray.h"

// helium
#include "helium/BaseGlobalDeviceState.h"
// anari
#include <anari/anari_cpp.hpp>
// optix
#include <helium/utility/TimeStamp.h>
#include <optix.h>
#include <optix_stubs.h>
// mdl
#ifdef USE_MDL
#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/imdl_compiler.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/iimage_api.h>
#endif // defined(USE_MDL)
// std
#include <vector>

#ifdef OPAQUE
#undef OPAQUE
#endif

constexpr int PAYLOAD_VALUES = 5;
constexpr int ATTRIBUTE_VALUES = 4;

#define OPTIX_CHECK_REPORT_MESSAGE(call)                                       \
  std::stringstream ss;                                                        \
  const char *res_str = optixGetErrorName(res);                                \
  ss << "Optix call (" << #call << ") failed with code " << res_str            \
     << " (line " << __LINE__ << ")\n";                                        \
  reportMessage(ANARI_SEVERITY_FATAL_ERROR, "%s", ss.str().c_str());

#define OPTIX_CHECK(call)                                                      \
  {                                                                            \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                                \
      OPTIX_CHECK_REPORT_MESSAGE(call)                                         \
    }                                                                          \
  }

#define OPTIX_CHECK_RETURN(call)                                               \
  {                                                                            \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                                \
      OPTIX_CHECK_REPORT_MESSAGE(call)                                         \
      return;                                                                  \
    }                                                                          \
  }

#define OPTIX_CHECK_RETURN_VALUE(call, x)                                      \
  {                                                                            \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                                \
      OPTIX_CHECK_REPORT_MESSAGE(call)                                         \
      return x;                                                                \
    }                                                                          \
  }

#define OPTIX_CHECK_OBJECT(call, obj)                                          \
  {                                                                            \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                                \
      std::stringstream ss;                                                    \
      const char *res_str = optixGetErrorName(res);                            \
      ss << "Optix call (" << #call << ") failed with code " << res_str        \
         << " (line " << __LINE__ << ")\n";                                    \
      obj->reportMessage(ANARI_SEVERITY_FATAL_ERROR, "%s", ss.str().c_str());  \
    }                                                                          \
  }

#define CUDA_SYNC_CHECK()                                                      \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t error = cudaGetLastError();                                    \
    if (error != cudaSuccess) {                                                \
      reportMessage(ANARI_SEVERITY_FATAL_ERROR,                                \
          "error (%s: line %d): %s\n",                                         \
          __FILE__,                                                            \
          __LINE__,                                                            \
          cudaGetErrorString(error));                                          \
    }                                                                          \
  }

#define CUDA_SYNC_CHECK_OBJECT(obj)                                            \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t error = cudaGetLastError();                                    \
    if (error != cudaSuccess) {                                                \
      obj->reportMessage(ANARI_SEVERITY_FATAL_ERROR,                           \
          "error (%s: line %d): %s\n",                                         \
          __FILE__,                                                            \
          __LINE__,                                                            \
          cudaGetErrorString(error));                                          \
    }                                                                          \
  }

#define VISRTX_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);                              \
  }

#define VISRTX_ANARI_TYPEFOR_DEFINITION(type)                                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_DEFINITION(type);                                              \
  }

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::box1, ANARI_FLOAT32_BOX1);

namespace visrtx {

struct Object;
struct MDL;

struct ptx_blob
{
  const unsigned char *ptr{nullptr};
  size_t size{0};
};

struct DeviceGlobalState : public helium::BaseGlobalDeviceState
{
  anari::Device anariDevice{nullptr};

  CUcontext cudaContext;
  CUstream stream;
  cudaDeviceProp deviceProps;

  OptixDeviceContext optixContext;

  struct RendererModules
  {
    OptixModule debug{nullptr};
    OptixModule raycast{nullptr};
    OptixModule ambientOcclusion{nullptr};
    OptixModule diffusePathTracer{nullptr};
    OptixModule directLight{nullptr};
    OptixModule test{nullptr};
    OptixModule mdl{nullptr};
  } rendererModules;

  struct IntersectionModules
  {
    OptixModule curveIntersector{nullptr};
    OptixModule customIntersectors{nullptr};
  } intersectionModules;

  struct MaterialModules
  {
    OptixModule matte{nullptr};
    OptixModule physicallyBased{nullptr};
  } materialShaders;

  struct ObjectUpdates
  {
    helium::TimeStamp lastBLASChange{0};
    helium::TimeStamp lastTLASChange{0};
    helium::TimeStamp lastMDLMaterialChange{0};
  } objectUpdates;

  DeferredArrayUploadBuffer uploadBuffer;

  struct DeviceObjectRegistry
  {
    DeviceObjectArray<SamplerGPUData> samplers;
    DeviceObjectArray<GeometryGPUData> geometries;
    DeviceObjectArray<MaterialGPUData> materials;
    DeviceObjectArray<SurfaceGPUData> surfaces;
    DeviceObjectArray<LightGPUData> lights;
    DeviceObjectArray<SpatialFieldGPUData> fields;
    DeviceObjectArray<VolumeGPUData> volumes;
  } registry;

  // MDL
#ifdef USE_MDL
  struct MDLContext
  {
  private:
    struct UuidHasher
    {
      std::size_t operator()(const mi::base::Uuid &uuid) const noexcept
      {
        return mi::base::uuid_hash32(uuid);
      }
    };

  public:
    mi::base::Handle<mi::neuraylib::INeuray> neuray;
    mi::base::Handle<mi::base::ILogger> logger;
    mi::base::Handle<mi::neuraylib::IMdl_compiler> mdlCompiler;
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdlConfiguration;
    mi::base::Handle<mi::neuraylib::IDatabase> database;
    mi::base::Handle<mi::neuraylib::IScope> globalScope;
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory;

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> executionContext;

    mi::base::Handle<mi::neuraylib::IMdl_backend> backendCudaPtx;
    mi::base::Handle<mi::neuraylib::IImage_api> imageApi;

    using TargetCodeCache = std::unordered_map<mi::base::Uuid, mi::base::Handle<mi::neuraylib::ITarget_code const>, UuidHasher>;

    /// Maps a compiled material hash to a target code object to avoid generation
    /// of duplicate code.
    TargetCodeCache targetCodeCache;
#if MI_PLATFORM_WINDOWS
    HMODULE dllHandle = nullptr;
#else
    void* dllHandle = {};
#endif
  } mdl;
#endif // defined(USE_MDL)
  // Helper methods //

  DeviceGlobalState(ANARIDevice d);
  ~DeviceGlobalState() override;
};

void buildOptixBVH(std::vector<OptixBuildInput> buildInput,
    DeviceBuffer &bvh,
    OptixTraversableHandle &traversable,
    box3 &bounds,
    Object *obj);

} // namespace visrtx
