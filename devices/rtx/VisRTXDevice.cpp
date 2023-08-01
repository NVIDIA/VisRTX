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

#include "anari/ext/debug/DebugObject.h"

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

namespace visrtx {

///////////////////////////////////////////////////////////////////////////////
// Generated function declarations ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

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

anari::debug_device::ObjectFactory *getDebugFactory();

const char **query_extensions();

///////////////////////////////////////////////////////////////////////////////
// Helper functions ///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename... Args>
inline T *createRegisteredObject(Args &&...args)
{
  return new T(std::forward<Args>(args)...);
}

template <typename HANDLE_T, typename OBJECT_T>
inline HANDLE_T finalizeHandleForAPI(OBJECT_T *o)
{
  auto *s = o->deviceState();
  s->commitBuffer.addObject(o);
  return (HANDLE_T)o;
}

template <typename OBJECT_T, typename HANDLE_T, typename... Args>
inline HANDLE_T createObjectForAPI(DeviceGlobalState *s, Args &&...args)
{
  auto o = createRegisteredObject<OBJECT_T>(s, std::forward<Args>(args)...);
  return finalizeHandleForAPI<HANDLE_T>(o);
}

///////////////////////////////////////////////////////////////////////////////
// VisRTXDevice definitions ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Data Arrays ////////////////////////////////////////////////////////////////

ANARIArray1D VisRTXDevice::newArray1D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems)
{
  initDevice();
  CUDADeviceScope ds(this);

  Array1DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems = numItems;

  if (anari::isObject(type))
    return createObjectForAPI<ObjectArray, ANARIArray1D>(deviceState(), md);
  else
    return createObjectForAPI<Array1D, ANARIArray1D>(deviceState(), md);
}

ANARIArray2D VisRTXDevice::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  initDevice();
  CUDADeviceScope ds(this);

  Array2DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;

  return createObjectForAPI<Array2D, ANARIArray2D>(deviceState(), md);
}

ANARIArray3D VisRTXDevice::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
  initDevice();
  CUDADeviceScope ds(this);

  Array3DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;
  md.numItems3 = numItems3;

  return createObjectForAPI<Array3D, ANARIArray3D>(deviceState(), md);
}

void *VisRTXDevice::mapArray(ANARIArray a)
{
  CUDADeviceScope ds(this);
  return helium::BaseDevice::mapArray(a);
}

void VisRTXDevice::unmapArray(ANARIArray a)
{
  CUDADeviceScope ds(this);
  helium::BaseDevice::unmapArray(a);
}

// Renderable Objects /////////////////////////////////////////////////////////

ANARILight VisRTXDevice::newLight(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return finalizeHandleForAPI<ANARILight>(
      Light::createInstance(subtype, deviceState()));
}

ANARICamera VisRTXDevice::newCamera(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return finalizeHandleForAPI<ANARICamera>(
      Camera::createInstance(subtype, deviceState()));
}

ANARIGeometry VisRTXDevice::newGeometry(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return finalizeHandleForAPI<ANARIGeometry>(
      Geometry::createInstance(subtype, deviceState()));
}

ANARISpatialField VisRTXDevice::newSpatialField(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return finalizeHandleForAPI<ANARISpatialField>(
      SpatialField::createInstance(subtype, deviceState()));
}

ANARISurface VisRTXDevice::newSurface()
{
  initDevice();
  CUDADeviceScope ds(this);
  return createObjectForAPI<Surface, ANARISurface>(deviceState());
}

ANARIVolume VisRTXDevice::newVolume(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return finalizeHandleForAPI<ANARIVolume>(
      Volume::createInstance(subtype, deviceState()));
}

// Surface Meta-Data //////////////////////////////////////////////////////////

ANARIMaterial VisRTXDevice::newMaterial(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return finalizeHandleForAPI<ANARIMaterial>(
      Material::createInstance(subtype, deviceState()));
}

ANARISampler VisRTXDevice::newSampler(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return finalizeHandleForAPI<ANARISampler>(
      Sampler::createInstance(subtype, deviceState()));
}

// Instancing /////////////////////////////////////////////////////////////////

ANARIGroup VisRTXDevice::newGroup()
{
  initDevice();
  CUDADeviceScope ds(this);
  return createObjectForAPI<Group, ANARIGroup>(deviceState());
}

ANARIInstance VisRTXDevice::newInstance()
{
  initDevice();
  CUDADeviceScope ds(this);
  return createObjectForAPI<Instance, ANARIInstance>(deviceState());
}

// Top-level Worlds ///////////////////////////////////////////////////////////

ANARIWorld VisRTXDevice::newWorld()
{
  initDevice();
  CUDADeviceScope ds(this);
  return createObjectForAPI<World, ANARIWorld>(deviceState());
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
  if (handleIsDevice(object)) {
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
    } else if (prop == "debugObjects" && type == ANARI_FUNCTION_POINTER) {
      helium::writeToVoidP(mem, getDebugFactory);
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
  } else {
    CUDADeviceScope ds(this);
    if (mask == ANARI_WAIT)
      flushCommitBuffer();
    return helium::referenceFromHandle(object).getProperty(
        name, type, mem, mask);
  }

  return 0;
}

// Frame Manipulation /////////////////////////////////////////////////////////

ANARIFrame VisRTXDevice::newFrame()
{
  initDevice();
  CUDADeviceScope ds(this);
  return createObjectForAPI<Frame, ANARIFrame>(deviceState());
}

const void *VisRTXDevice::frameBufferMap(ANARIFrame f,
    const char *channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  CUDADeviceScope ds(this);
  return helium::BaseDevice::frameBufferMap(
      f, channel, width, height, pixelType);
}

// Frame Rendering ////////////////////////////////////////////////////////////

ANARIRenderer VisRTXDevice::newRenderer(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return finalizeHandleForAPI<ANARIRenderer>(
      Renderer::createInstance(subtype, deviceState()));
}

void VisRTXDevice::renderFrame(ANARIFrame f)
{
  CUDADeviceScope ds(this);
  helium::BaseDevice::renderFrame(f);
}

int VisRTXDevice::frameReady(ANARIFrame f, ANARIWaitMask m)
{
  CUDADeviceScope ds(this);
  return helium::BaseDevice::frameReady(f, m);
}

void VisRTXDevice::discardFrame(ANARIFrame f)
{
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
  if (m_state.get() == nullptr)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "destroying VisRTX device");

  auto &state = *deviceState();

  clearCommitBuffer();
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

  if (Frame::objectCount() != 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked frames",
        Frame::objectCount());
  }
  if (Camera::objectCount() != 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked cameras",
        Camera::objectCount());
  }
  if (Renderer::objectCount() != 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked renderers",
        Renderer::objectCount());
  }
  if (World::objectCount() != 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked worlds",
        World::objectCount());
  }
  if (Instance::objectCount() != 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked instances",
        Instance::objectCount());
  }
  if (Group::objectCount() != 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked groups",
        Group::objectCount());
  }

  auto reportLeakedRegistryValues = [&](const char *type, auto &r) {
    if (r.empty())
      return;

    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked %s objects",
        r.size(),
        type);

    for (int i = 0; i < r.capacity(); i++) {
      auto *obj = (Object *)r.hostObject(i);
      if (!obj)
        continue;
      auto name = obj->getParamString("name", "<no name>");
      reportMessage(ANARI_SEVERITY_WARNING,
          "    leaked %s (%p) | ref counts [%zu, %zu] | name '%s'",
          type,
          obj,
          obj->useCount(helium::RefType::PUBLIC),
          obj->useCount(helium::RefType::INTERNAL),
          name.c_str());
    }
  };

  reportLeakedRegistryValues("light", state.registry.lights);
  reportLeakedRegistryValues("surface", state.registry.surfaces);
  reportLeakedRegistryValues("volume", state.registry.volumes);
  reportLeakedRegistryValues("geometry", state.registry.geometries);
  reportLeakedRegistryValues("material", state.registry.materials);
  reportLeakedRegistryValues("spatial field", state.registry.fields);
  reportLeakedRegistryValues("sampler", state.registry.samplers);

  if (Array::objectCount() != 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked arrays",
        Array::objectCount());
  }
}

void VisRTXDevice::initDevice()
{
  if (m_initialized)
    return;

  auto &state = *deviceState();

  if (!m_eagerInit)
    deviceCommitParameters();

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing VisRTX device", this);

  cudaFree(nullptr);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0) {
    reportMessage(ANARI_SEVERITY_FATAL_ERROR, "no CUDA capable devices found!");
    m_state.reset();
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
        "error querying current context: error code '%i'",
        cuRes);
    return;
  }

  auto context_log_cb = [](unsigned int level,
                            const char *tag,
                            const char *message,
                            void *_device) {
    auto *device = (VisRTXDevice *)_device;
    auto severity =
        level <= 2 ? ANARI_SEVERITY_FATAL_ERROR : ANARI_SEVERITY_INFO;
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

  auto init_module = [&](OptixModule &module, unsigned char *ptx) {
    const std::string ptxCode = (const char *)ptx;

    std::string log(2048, '\n');
    size_t sizeof_log = log.size();

    OPTIX_CHECK(optixModuleCreateFromPTX(state.optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptxCode.c_str(),
        ptxCode.size(),
        log.data(),
        &sizeof_log,
        &module));

    if (sizeof_log > 1)
      reportMessage(ANARI_SEVERITY_DEBUG, "PTX Compile Log:\n%s", log.data());
  };

  reportMessage(ANARI_SEVERITY_DEBUG, "Compiling 'debug' renderer");
  init_module(state.rendererModules.debug, Debug::ptx());
  reportMessage(ANARI_SEVERITY_DEBUG, "Compiling 'raycast' renderer");
  init_module(state.rendererModules.raycast, Raycast::ptx());
  reportMessage(ANARI_SEVERITY_DEBUG, "Compiling 'ao' renderer");
  init_module(state.rendererModules.ambientOcclusion, AmbientOcclusion::ptx());
  reportMessage(ANARI_SEVERITY_DEBUG, "Compiling 'dpt' renderer");
  init_module(
      state.rendererModules.diffusePathTracer, DiffusePathTracer::ptx());
  reportMessage(ANARI_SEVERITY_DEBUG, "Compiling 'scivis' renderer");
  init_module(state.rendererModules.scivis, SciVis::ptx());
  reportMessage(ANARI_SEVERITY_DEBUG, "Compiling 'test' renderer");
  init_module(state.rendererModules.test, Test::ptx());

  reportMessage(ANARI_SEVERITY_DEBUG, "Compiling custom intersectors");
  init_module(state.intersectionModules.customIntersectors, intersection_ptx());

  OptixBuiltinISOptions builtinISOptions = {};
  builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
  builtinISOptions.usesMotionBlur = 0;
  OPTIX_CHECK(optixBuiltinISModuleGet(state.optixContext,
      &moduleCompileOptions,
      &pipelineCompileOptions,
      &builtinISOptions,
      &state.intersectionModules.curveIntersector));

  m_initialized = true;
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
    initDevice();
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
