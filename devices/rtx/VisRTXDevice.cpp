/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "VisRTXDevice.h"

#include "array/Array1D.h"
#include "array/Array2D.h"
#include "array/Array3D.h"
#include "array/ObjectArray.h"
#include "camera/Camera.h"
#include "frame/Frame.h"
#include "renderer/Renderer.h"
#include "scene/World.h"
#include "scene/surface/material/sampler/Sampler.h"
#include "scene/volume/spatial_field/SpatialField.h"

// PTX //

// renderers
#include "renderer/AmbientOcclusion.h"
#include "renderer/Debug.h"
#include "renderer/DiffusePathTracer.h"
#include "renderer/Raycast.h"
#include "renderer/SciVis.h"
#include "renderer/Test.h"

// std
#include <future>

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

  state.commitBufferClear();
  state.uploadBuffer.clear();

  CUDA_SYNC_CHECK();

  optixModuleDestroy(state.rendererModules.debug);
  optixModuleDestroy(state.rendererModules.raycast);
  optixModuleDestroy(state.rendererModules.ambientOcclusion);
  optixModuleDestroy(state.rendererModules.diffusePathTracer);
  optixModuleDestroy(state.rendererModules.scivis);
  optixModuleDestroy(state.rendererModules.test);

  optixModuleDestroy(state.intersectionModules.customIntersectors);

  optixDeviceContextDestroy(state.optixContext);
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

  initOptix();

  return m_initStatus == DeviceInitStatus::SUCCESS;
}

void VisRTXDevice::deviceCommitParameters()
{
  helium::BaseDevice::deviceCommitParameters();
  m_eagerInit = getParam<bool>("forceInit", false);
  m_desiredGpuID = getParam<int>("cudaDevice", 0);
  if (m_gpuID >= 0 && m_desiredGpuID != m_gpuID) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "visrtx was already initialized to use GPU %i"
        ": new device number %i is ignored.",
        m_gpuID,
        m_desiredGpuID);
  }

  if (m_eagerInit)
    initOptix();
}

int VisRTXDevice::deviceGetProperty(
    const char *name, ANARIDataType type, void *mem, uint64_t size)
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
  } else if (prop == "feature" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, query_extensions());
    return 1;
  } else if (prop == "subtypes.renderer" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, query_object_types(ANARI_RENDERER));
    return 1;
  } else if (prop == "visrtx" && type == ANARI_BOOL) {
    helium::writeToVoidP(mem, true);
    return 1;
  }
  return 0;
}

void VisRTXDevice::initOptix()
{
  if (m_initStatus != DeviceInitStatus::UNINITIALIZED)
    return;

  auto &state = *deviceState();

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing VisRTX device", this);

  int numDevices = 0;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0) {
    reportMessage(ANARI_SEVERITY_FATAL_ERROR, "no CUDA capable devices found!");
    m_initStatus = DeviceInitStatus::FAILURE;
    return;
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

  OPTIX_CHECK(optixInit());
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
    m_initStatus = DeviceInitStatus::FAILURE;
    return;
  }

  auto context_log_cb = [](unsigned int level,
                            const char *tag,
                            const char *message,
                            void *_device) {
    auto *device = (VisRTXDevice *)_device;
    auto severity =
        level <= 2 ? ANARI_SEVERITY_FATAL_ERROR : ANARI_SEVERITY_DEBUG;
    device->reportMessage(
        severity, "OptiX message [%u][%s]:\n%s", level, tag, message);
  };

  OptixDeviceContextOptions options{};
  options.logCallbackFunction = context_log_cb;
  options.logCallbackData = this;
  options.logCallbackLevel = 4;

  OPTIX_CHECK(optixDeviceContextCreate(
      state.cudaContext, &options, &state.optixContext));

  // Create OptiX modules //

  OptixModuleCompileOptions moduleCompileOptions = {};
  moduleCompileOptions.maxRegisterCount =
      OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

  auto pipelineCompileOptions = makeVisRTXOptixPipelineCompileOptions();

  auto init_module = [&](OptixModule &module,
                         unsigned char *ptx,
                         const char *name) -> std::future<void> {
    auto f = std::async([&]() {
      reportMessage(ANARI_SEVERITY_INFO, "Compiling OptiX module: %s", name);

      const std::string ptxCode = (const char *)ptx;

      std::string log(2048, '\n');
      size_t sizeof_log = log.size();

#if OPTIX_VERSION < 70700
      OPTIX_CHECK(optixModuleCreateFromPTX(state.optixContext,
          &moduleCompileOptions,
          &pipelineCompileOptions,
          ptxCode.c_str(),
          ptxCode.size(),
          log.data(),
          &sizeof_log,
          &module));
#else
      OPTIX_CHECK(optixModuleCreate(state.optixContext,
          &moduleCompileOptions,
          &pipelineCompileOptions,
          ptxCode.c_str(),
          ptxCode.size(),
          log.data(),
          &sizeof_log,
          &module));
#endif

      if (sizeof_log > 1)
        reportMessage(ANARI_SEVERITY_DEBUG, "PTX Compile Log:\n%s", log.data());
    });
#ifndef VISRTX_PARALLEL_MODULE_BUILD
    f.wait();
#endif
    return f;
  };

  std::vector<std::future<void>> compileTasks;

  compileTasks.push_back(init_module(
      state.rendererModules.debug, Debug::ptx(), "'debug' renderer"));
  compileTasks.push_back(init_module(
      state.rendererModules.raycast, Raycast::ptx(), "'raycast' renderer"));
  compileTasks.push_back(init_module(state.rendererModules.ambientOcclusion,
      AmbientOcclusion::ptx(),
      "'ao' renderer"));
  compileTasks.push_back(init_module(state.rendererModules.diffusePathTracer,
      DiffusePathTracer::ptx(),
      "'dpt' renderer"));
  compileTasks.push_back(init_module(
      state.rendererModules.scivis, SciVis::ptx(), "'scivis' renderer"));
  compileTasks.push_back(
      init_module(state.rendererModules.test, Test::ptx(), "'test' renderer"));

  compileTasks.push_back(
      init_module(state.intersectionModules.customIntersectors,
          intersection_ptx(),
          "custom intersectors"));

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

  m_initStatus = DeviceInitStatus::SUCCESS;
}

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
