/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "anari_library_visrtx_export.h"

// std
#include <cstdlib>
// nvml
#include <nvml.h>

#include <anari/frontend/anari_enums.h>
#include <optix_types.h>
#include "VisRTXDevice.h"

#include "array/Array1D.h"
#include "array/Array2D.h"
#include "array/Array3D.h"
#include "array/ObjectArray.h"
#include "camera/Camera.h"
#include "frame/Frame.h"
#include "optix_visrtx.h"
#include "renderer/Renderer.h"
#include "sampler/Sampler.h"
#include "spatial_field/SpatialField.h"
#include "world/World.h"

// PTX //

// renderers
#include "renderer/Fast.h"
#include "renderer/Debug.h"
#include "renderer/Interactive.h"
#include "renderer/Quality.h"
#include "renderer/Test.h"

// materials
#include "material/shaders/MatteShader.h"
#include "material/shaders/PhysicallyBasedShader.h"

// spatial field samplers
#include "spatial_field/NvdbRectilinearSampler.h"
#include "spatial_field/NvdbRegularSampler.h"
#include "spatial_field/StructuredRectilinearSampler.h"
#include "spatial_field/StructuredRegularSampler.h"
// custom field sampler (from devices/visrtx)
#include "spatial_field/CustomFieldSampler.h"

// MDL
#ifdef USE_MDL
#include "libmdl/Core.h"
#endif // defined(USE_MDL)

// std
#include <filesystem>
#include <future>
#include <string>
#include <vector>

namespace visrtx {

const char **query_object_types(ANARIDataType type);

const void *query_object_info(ANARIDataType type,
    const char *subtype,
    const char *infoName,
    ANARIDataType infoType);

const void *query_param_info(ANARIDataType type,
    const char *subtype,
    const char *paramName,
    ANARIDataType paramType,
    const char *infoName,
    ANARIDataType infoType);

const char **query_extensions();

///////////////////////////////////////////////////////////////////////////////
// VisRTXDevice definitions ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Data Arrays ////////////////////////////////////////////////////////////////

void *VisRTXDevice::mapArray(ANARIArray a)
{
  if (!initDevice())
    return nullptr;
  CUDADeviceScope ds(this);
  return helium::BaseDevice::mapArray(a);
}

void VisRTXDevice::unmapArray(ANARIArray a)
{
  if (!initDevice())
    return;
  CUDADeviceScope ds(this);
  helium::BaseDevice::unmapArray(a);
}

// API Objects ////////////////////////////////////////////////////////////////

ANARIArray1D VisRTXDevice::newArray1D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems)
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);

  Array1DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems = numItems;

  if (anari::isObject(type))
    return (ANARIArray1D) new ObjectArray(deviceState(), md);
  else
    return (ANARIArray1D) new Array1D(deviceState(), md);
}

ANARIArray2D VisRTXDevice::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);

  Array2DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;

  return (ANARIArray2D) new Array2D(deviceState(), md);
}

ANARIArray3D VisRTXDevice::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);

  Array3DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;
  md.numItems3 = numItems3;

  return (ANARIArray3D) new Array3D(deviceState(), md);
}

ANARICamera VisRTXDevice::newCamera(const char *subtype)
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARICamera)Camera::createInstance(subtype, deviceState());
}

ANARIFrame VisRTXDevice::newFrame()
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARIFrame) new Frame(deviceState());
}

ANARIGeometry VisRTXDevice::newGeometry(const char *subtype)
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARIGeometry)Geometry::createInstance(subtype, deviceState());
}

ANARIGroup VisRTXDevice::newGroup()
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARIGroup) new Group(deviceState());
}

ANARIInstance VisRTXDevice::newInstance(const char *type)
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARIInstance) new Instance(deviceState());
}

ANARILight VisRTXDevice::newLight(const char *subtype)
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARILight)Light::createInstance(subtype, deviceState());
}

ANARIMaterial VisRTXDevice::newMaterial(const char *subtype)
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARIMaterial)Material::createInstance(subtype, deviceState());
}

ANARIRenderer VisRTXDevice::newRenderer(const char *subtype)
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARIRenderer)Renderer::createInstance(subtype, deviceState());
}

ANARISampler VisRTXDevice::newSampler(const char *subtype)
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARISampler)Sampler::createInstance(subtype, deviceState());
}

ANARISpatialField VisRTXDevice::newSpatialField(const char *subtype)
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARISpatialField)SpatialField::createInstance(
      subtype, deviceState());
}

ANARISurface VisRTXDevice::newSurface()
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARISurface) new Surface(deviceState());
}

ANARIVolume VisRTXDevice::newVolume(const char *subtype)
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARIVolume)Volume::createInstance(subtype, deviceState());
}

ANARIWorld VisRTXDevice::newWorld()
{
  if (!initDevice())
    return {};
  CUDADeviceScope ds(this);
  return (ANARIWorld) new World(deviceState());
}

// Query functions ////////////////////////////////////////////////////////////

const char **VisRTXDevice::getObjectSubtypes(ANARIDataType objectType)
{
  return query_object_types(objectType);
}

const void *VisRTXDevice::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  return query_object_info(objectType, objectSubtype, infoName, infoType);
}

const void *VisRTXDevice::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  return query_param_info(objectType,
      objectSubtype,
      parameterName,
      parameterType,
      infoName,
      infoType);
}

// Object + Parameter Lifetime Management /////////////////////////////////////

int VisRTXDevice::getProperty(ANARIObject object,
    const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    uint32_t mask)
{
  if (!initDevice())
    return 0;
  CUDADeviceScope ds(this);
  return helium::BaseDevice::getProperty(object, name, type, mem, size, mask);
}

// Frame Manipulation /////////////////////////////////////////////////////////

const void *VisRTXDevice::frameBufferMap(ANARIFrame f,
    const char *channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  if (!initDevice()) {
    *width = 0;
    *height = 0;
    *pixelType = ANARI_UNKNOWN;
    return nullptr;
  }

  return helium::BaseDevice::frameBufferMap(
      f, channel, width, height, pixelType);
}

// Frame Rendering ////////////////////////////////////////////////////////////

void VisRTXDevice::renderFrame(ANARIFrame f)
{
  if (!initDevice())
    return;
  CUDADeviceScope ds(this);
  helium::BaseDevice::renderFrame(f);
}

int VisRTXDevice::frameReady(ANARIFrame f, ANARIWaitMask m)
{
  if (!initDevice())
    return 1;
  CUDADeviceScope ds(this);
  return helium::BaseDevice::frameReady(f, m);
}

void VisRTXDevice::discardFrame(ANARIFrame f)
{
  if (!initDevice())
    return;
  CUDADeviceScope ds(this);
  return helium::BaseDevice::discardFrame(f);
}

// Other VisRTXDevice definitions /////////////////////////////////////////////

VisRTXDevice::VisRTXDevice(ANARIStatusCallback cb, const void *ptr)
    : helium::BaseDevice(cb, ptr)
{
  m_state = std::make_unique<DeviceGlobalState>(this_device());
  helium::BaseDevice::deviceCommitParameters();
}

VisRTXDevice::VisRTXDevice(ANARILibrary l) : helium::BaseDevice(l)
{
  m_state = std::make_unique<DeviceGlobalState>(this_device());
  helium::BaseDevice::deviceCommitParameters();
}

VisRTXDevice::~VisRTXDevice()
{
  reportMessage(ANARI_SEVERITY_DEBUG, "destroying VisRTX device");

  if (m_initStatus != DeviceInitStatus::SUCCESS)
    return;

  auto &state = *deviceState();

#ifdef USE_MDL
  if (m_mdlInitStatus == DeviceInitStatus::SUCCESS) {
    state.mdl.reset();
    m_mdlInitStatus = DeviceInitStatus::UNINITIALIZED;
  }
#endif // defined(USE_MDL)

  state.commitBuffer.clear();
  state.uploadBuffer.clear();

  CUDA_SYNC_CHECK();

  optixModuleDestroy(state.rendererModules.debug);
  optixModuleDestroy(state.rendererModules.fast);
  optixModuleDestroy(state.rendererModules.quality);
  optixModuleDestroy(state.rendererModules.interactive);
#ifdef USE_MDL
  optixModuleDestroy(state.rendererModules.mdl);
#endif // defined(USE_MDL)
  optixModuleDestroy(state.rendererModules.test);

  optixModuleDestroy(state.intersectionModules.customIntersectors);

  optixDeviceContextDestroy(state.optixContext);

  cudaStreamDestroy(state.stream);
}

bool VisRTXDevice::initDevice()
{
  if (m_initStatus == DeviceInitStatus::SUCCESS)
    return true;
  else if (m_initStatus == DeviceInitStatus::FAILURE) {
    reportMessage(ANARI_SEVERITY_ERROR, "device failed to initialized");
    return false;
  }

  auto deviceLock = scopeLockObject();

  // recheck in case more than one thread tried to init
  if (m_initStatus != DeviceInitStatus::UNINITIALIZED)
    return m_initStatus == DeviceInitStatus::SUCCESS;

  if (!m_eagerInit)
    deviceCommitParameters();

  m_initStatus = initOptix();
#ifdef USE_MDL
  m_mdlInitStatus = initMDL();
#endif // defined(USE_MDL)
  return m_initStatus == DeviceInitStatus::SUCCESS;
}

void VisRTXDevice::deviceCommitParameters()
{
  helium::BaseDevice::deviceCommitParameters();
  m_eagerInit = getParam<bool>("forceInit", false);
  m_desiredGpuID = getParam<int>("cudaDevice", -1);

  if (m_desiredGpuID < 0) {
    if (auto *env = std::getenv("VISRTX_CUDA_DEVICE"); env) {
      reportMessage(ANARI_SEVERITY_DEBUG,
          "overriding cudaDevice param with VISRTX_CUDA_DEVICE env var: %s",
          env);
      m_desiredGpuID = std::atoi(env);
    } else {
      m_desiredGpuID = 0;
    }
  }

  if (m_gpuID >= 0 && m_desiredGpuID != m_gpuID) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "visrtx was already initialized to use GPU %i"
        ": new device number %i is ignored.",
        m_gpuID,
        m_desiredGpuID);
  }

  if (m_eagerInit && m_initStatus == DeviceInitStatus::UNINITIALIZED) {
    reportMessage(ANARI_SEVERITY_DEBUG, "eagerly initializing device");
    m_initStatus = initOptix();
#ifdef USE_MDL
    m_mdlInitStatus = initMDL();
#endif // defined(USE_MDL)
  }

#ifdef USE_MDL
  if (m_mdlInitStatus == DeviceInitStatus::SUCCESS) {
    auto paths = getParamString("mdlSearchPaths", "");
    auto mdlSearchPaths = std::vector<std::filesystem::path>{};

#if defined(WIN32)
    constexpr char sep = ';';
#else
    constexpr char sep = ':';
#endif

    for (auto it = cbegin(paths);;) {
      while (it != cend(paths) && *it == sep)
        ++it;
      if (it == cend(paths))
        break;
      auto endOfPathIt = std::find(it, cend(paths), sep);
      mdlSearchPaths.emplace_back(it, endOfPathIt);
      it = endOfPathIt;
    }
    deviceState()->mdl->core.setMdlSearchPaths(mdlSearchPaths);

    paths = getParamString("mdlResourceSearchPaths", "");
    mdlSearchPaths = std::vector<std::filesystem::path>{};

    for (auto it = cbegin(paths);;) {
      while (it != cend(paths) && *it == sep)
        ++it;
      if (it == cend(paths))
        break;
      auto endOfPathIt = std::find(it, cend(paths), sep);
      mdlSearchPaths.emplace_back(it, endOfPathIt);
      it = endOfPathIt;
    }
    deviceState()->mdl->core.setMdlResourceSearchPaths(mdlSearchPaths);
  }
#endif // defined(USE_MDL)
}

int VisRTXDevice::deviceGetProperty(const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    uint32_t /*flags*/)
{
  std::string_view prop = name;
  if (prop == "version" && type == ANARI_INT32) {
    int version = VISRTX_VERSION_MAJOR * 10000 + VISRTX_VERSION_MINOR * 100
        + VISRTX_VERSION_PATCH;
    helium::writeToVoidP(mem, version);
    return 1;
  } else if (prop == "version.major" && type == ANARI_INT32) {
    helium::writeToVoidP(mem, VISRTX_VERSION_MAJOR);
    return 1;
  } else if (prop == "version.minor" && type == ANARI_INT32) {
    helium::writeToVoidP(mem, VISRTX_VERSION_MINOR);
    return 1;
  } else if (prop == "version.patch" && type == ANARI_INT32) {
    helium::writeToVoidP(mem, VISRTX_VERSION_PATCH);
    return 1;
  } else if (prop == "extension" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, query_extensions());
    return 1;
  } else if (prop == "subtypes.renderer" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, query_object_types(ANARI_RENDERER));
    return 1;
  } else if (prop == "visrtx" && type == ANARI_BOOL) {
    helium::writeToVoidP(mem, true);
    return 1;
  } else if (prop == "geometryMaxIndex" && type == ANARI_UINT64) {
    helium::writeToVoidP(mem, uint64_t(std::numeric_limits<uint32_t>::max()));
    return 1;
  }
  return 0;
}

DeviceInitStatus VisRTXDevice::initOptix()
{
  if (m_initStatus != DeviceInitStatus::UNINITIALIZED)
    return m_initStatus;

  auto &state = *deviceState();

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing VisRTX device", this);

  int numDevices = 0;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0) {
    reportMessage(ANARI_SEVERITY_FATAL_ERROR, "no CUDA capable devices found!");
    return DeviceInitStatus::FAILURE;
  }

  if (m_desiredGpuID >= numDevices) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "desired GPU selection (%i) is not a valid ID as only "
        "%i GPUs were found, defaulting to GPU 0",
        m_desiredGpuID,
        numDevices);
    m_desiredGpuID = 0;
  }
  m_gpuID = m_desiredGpuID;

  {
    int cudaVersion = 0;
    cudaDriverGetVersion(&cudaVersion);
    int major = cudaVersion / 1000;
    int minor = (cudaVersion % 1000) / 10;
    reportMessage(
        ANARI_SEVERITY_DEBUG, "VisRTX using CUDA %i.%i", major, minor);
  }

  {
    char driverVersion[80]; // Buffer to store driver version

    nvmlInit();
    nvmlSystemGetDriverVersion(driverVersion, sizeof(driverVersion));
    reportMessage(
        ANARI_SEVERITY_DEBUG, "VisRTX running on driver %s", driverVersion);
    int driverVersionMajor = 0;
    std::sscanf(driverVersion, "%i.", &driverVersionMajor);
    if (OPTIX_VERSION >= 90000 && driverVersionMajor < 570) {
      reportMessage(ANARI_SEVERITY_FATAL_ERROR,
          "Using OptiX 9.0+ requires NVIDIA driver version 570 or higher, "
          "found %i.x",
          driverVersionMajor);
      return DeviceInitStatus::FAILURE;
    } else if (OPTIX_VERSION == 80100 && driverVersionMajor < 555) {
      reportMessage(ANARI_SEVERITY_FATAL_ERROR,
          "Using OptiX 8.1 requires NVIDIA driver version 555 or higher, "
          "found %i.x",
          driverVersionMajor);
      return DeviceInitStatus::FAILURE;
    } else if (OPTIX_VERSION == 80000 && driverVersionMajor < 535) {
      reportMessage(ANARI_SEVERITY_FATAL_ERROR,
          "Using OptiX 8.0 requires NVIDIA driver version 535 or higher, "
          "found %i.x",
          driverVersionMajor);
      return DeviceInitStatus::FAILURE;
    } else if (OPTIX_VERSION == 70700 && driverVersionMajor < 530) {
      reportMessage(ANARI_SEVERITY_FATAL_ERROR,
          "Using OptiX 7.7 requires NVIDIA driver version 530 or higher, "
          "found %i.x",
          driverVersionMajor);
      return DeviceInitStatus::FAILURE;
    }
    nvmlShutdown();
  }

  OPTIX_CHECK_RETURN_VALUE(optixInit(), DeviceInitStatus::FAILURE);
  setCUDADevice();
  cudaStreamCreate(&state.stream);

  cudaGetDeviceProperties(&state.deviceProps, m_gpuID);
  reportMessage(ANARI_SEVERITY_DEBUG,
      "running on GPU %i (%s)\n",
      m_gpuID,
      state.deviceProps.name);

  CUresult cuRes = cuCtxGetCurrent(&state.cudaContext);
  if (cuRes != CUDA_SUCCESS) {
    reportMessage(ANARI_SEVERITY_FATAL_ERROR,
        "error querying current CUDA context: error code '%i'",
        cuRes);
    return DeviceInitStatus::FAILURE;
  }

  auto context_log_cb = [](unsigned int level,
                            const char *tag,
                            const char *message,
                            void *_device) {
    auto *device = (VisRTXDevice *)_device;
    ANARIStatusSeverity severity = ANARI_SEVERITY_DEBUG;
    switch (level) {
    case 0:
    case 1:
      severity = ANARI_SEVERITY_FATAL_ERROR;
      break;
    case 2:
      severity = ANARI_SEVERITY_ERROR;
      break;
    case 3:
      severity = ANARI_SEVERITY_WARNING;
      break;
    case 4:
    default:
      severity = ANARI_SEVERITY_DEBUG;
    }
    device->reportMessage(severity, "OptiX message {%s} -- %s", tag, message);
  };

  OptixDeviceContextOptions options{};
  options.logCallbackFunction = context_log_cb;
  options.logCallbackData = this;
  options.logCallbackLevel = 4;

  OPTIX_CHECK_RETURN_VALUE(
      optixDeviceContextCreate(
          state.cudaContext, &options, &state.optixContext),
      DeviceInitStatus::FAILURE);

  // Create OptiX modules //

  OptixModuleCompileOptions moduleCompileOptions = {};
  moduleCompileOptions.maxRegisterCount =
      OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
  moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
#else
  moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
  moduleCompileOptions.debugLevel =
      OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL; // Could be FULL if -G can be enabled
                                         // at compile time
#endif

  auto pipelineCompileOptions = makeVisRTXOptixPipelineCompileOptions();

  auto init_module = [&](OptixModule *module,
                         ptx_blob ptx,
                         const char *name) -> std::future<void> {
    // Make sure that we capture init_module function parameters by value
    // as those will be out of scope as soon as we exit the function, thereby
    // creating dandling references to those parameters if we do not capture by
    // value.
    auto f = std::async([&, module, ptx, name]() {
      reportMessage(ANARI_SEVERITY_INFO, "Compiling OptiX module: %s", name);

      std::string log(2048, '\n');
      size_t sizeof_log = log.size();

#if OPTIX_VERSION < 70700
      OPTIX_CHECK(optixModuleCreateFromPTX(state.optixContext,
          &moduleCompileOptions,
          &pipelineCompileOptions,
          (const char *)ptx.ptr,
          ptx.size,
          log.data(),
          &sizeof_log,
          &module));
#else
      OPTIX_CHECK(optixModuleCreate(state.optixContext,
          &moduleCompileOptions,
          &pipelineCompileOptions,
          (const char *)ptx.ptr,
          ptx.size,
          log.data(),
          &sizeof_log,
          module));
#endif

      if (sizeof_log > 1)
        reportMessage(ANARI_SEVERITY_DEBUG, "PTX Compile Log:\n%s", log.data());
    });
#ifndef VISRTX_PARALLEL_MODULE_BUILD
    f.wait();
#endif
    return f;
  };

  auto compileTasks = std::array{
      init_module(
          &state.rendererModules.debug, Debug::ptx(), "'debug' renderer"),
      init_module(
          &state.rendererModules.fast, Fast::ptx(), "'fast' renderer"),
      init_module(
          &state.rendererModules.quality, Quality::ptx(), "'quality' renderer"),
      init_module(&state.rendererModules.interactive,
          Interactive::ptx(),
          "'interactive' renderer"),
      init_module(&state.rendererModules.test, Test::ptx(), "'test' renderer"),
      init_module(&state.intersectionModules.customIntersectors,
          intersection_ptx(),
          "custom intersectors"),
      init_module(
          &state.materialShaders.matte, MatteShader::ptx(), "'matte' shader"),
      init_module(&state.materialShaders.physicallyBased,
          PhysicallyBasedShader::ptx(),
          "'physicallyBased' shader"),
      init_module(&state.fieldSamplers.structuredRegular,
          StructuredRegularSampler::ptx(),
          "'structuredRegular' field sampler"),
      init_module(&state.fieldSamplers.nvdb,
          NvdbRegularSampler::ptx(),
          "'nvdb' field sampler"),
      init_module(&state.fieldSamplers.structuredRectilinear,
          StructuredRectilinearSampler::ptx(),
          "'structuredRectilinear' field sampler"),
      init_module(&state.fieldSamplers.nvdbRectilinear,
          NvdbRectilinearSampler::ptx(),
          "'nanovdbRectilinear' field sampler"),
      // Custom field sampler module (from devices/visrtx)
      init_module(&state.fieldSamplers.customField,
          CustomFieldSampler::ptx(),
          "'customField' sampler"),
  };

  for (auto &f : compileTasks)
    f.wait();

  OptixBuiltinISOptions builtinISOptions = {};
  builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
  builtinISOptions.usesMotionBlur = 0;
  OPTIX_CHECK(optixBuiltinISModuleGet(state.optixContext,
      &moduleCompileOptions,
      &pipelineCompileOptions,
      &builtinISOptions,
      &state.intersectionModules.curveIntersector));

  return DeviceInitStatus::SUCCESS;
}

#ifdef USE_MDL
DeviceInitStatus VisRTXDevice::initMDL()
{
  if (m_mdlInitStatus != DeviceInitStatus::UNINITIALIZED)
    return m_mdlInitStatus;

  auto ineuray = static_cast<mi::neuraylib::INeuray *>(
      getParam<void *>("ineuray", nullptr));
  try {
    auto deviceGlobalState = deviceState();
    if (ineuray) {
      deviceGlobalState->mdl.emplace(deviceGlobalState, ineuray);
    } else {
      deviceGlobalState->mdl.emplace(deviceGlobalState);
    }
    ineuray = deviceGlobalState->mdl->core.getINeuray();
  } catch (const std::exception &e) {
    reportMessage(
        ANARI_SEVERITY_ERROR, "Failed initializing VisRTX MDL support.");
    return DeviceInitStatus::FAILURE;
  }

  return DeviceInitStatus::SUCCESS;
}
#endif // defined(USE_MDL)

void VisRTXDevice::setCUDADevice()
{
  cudaGetDevice(&m_appGpuID);
  cudaSetDevice(m_gpuID);
}

void VisRTXDevice::revertCUDADevice()
{
  cudaSetDevice(m_appGpuID);
}

DeviceGlobalState *VisRTXDevice::deviceState() const
{
  return (DeviceGlobalState *)helium::BaseDevice::m_state.get();
}

VisRTXDevice::CUDADeviceScope::CUDADeviceScope(VisRTXDevice *d) : m_device(d)
{
  m_device->setCUDADevice();
}

VisRTXDevice::CUDADeviceScope::~CUDADeviceScope()
{
  m_device->revertCUDADevice();
}

} // namespace visrtx
