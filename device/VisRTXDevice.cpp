/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "VisRTXDevice.h"

#include "anari/detail/Helpers.h"
#include "anari/detail/Library.h"

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
// std
#include <chrono>
#include <cstdarg>
#include <exception>
#include <functional>
#include <limits>
#include <thread>

// PTX //

// renderers
#include "renderer/AmbientOcclusion.h"
#include "renderer/Debug.h"
#include "renderer/DiffusePathTracer.h"
#include "renderer/Raycast.h"
#include "renderer/SciVis.h"
// geometry
#include "scene/surface/geometry/Spheres.h"

namespace visrtx {

///////////////////////////////////////////////////////////////////////////////
// Helper functions ///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void writeToVoidP(void *_p, T v)
{
  T *p = (T *)_p;
  *p = v;
}

template <typename T, typename... Args>
inline T *createRegisteredObject(Args &&...args)
{
  return new T(std::forward<Args>(args)...);
}

template <typename HANDLE_T, typename OBJECT_T>
inline HANDLE_T getHandleForAPI(OBJECT_T *object)
{
  return (HANDLE_T)object;
}

template <typename HANDLE_T, typename OBJECT_T>
inline HANDLE_T completeHandle(OBJECT_T *o)
{
  o->setObjectType(anari::ANARITypeFor<HANDLE_T>::value);
  o->markUpdated();
  o->deviceState()->commitBuffer.addObject(o);
  return getHandleForAPI<HANDLE_T>(o);
}

template <typename OBJECT_T, typename HANDLE_T, typename... Args>
inline HANDLE_T createObjectForAPI(DeviceGlobalState *s, Args &&...args)
{
  auto o = createRegisteredObject<OBJECT_T>(std::forward<Args>(args)...);
  o->setDeviceState(s);
  if constexpr (std::is_same_v<OBJECT_T, Surface>)
    o->setRegistry(s->registry.surfaces);
  return completeHandle<HANDLE_T>(o);
}

template <typename OBJECT_T = Object, typename HANDLE_T = ANARIObject>
inline OBJECT_T &referenceFromHandle(HANDLE_T handle)
{
  return *((OBJECT_T *)handle);
}

#define declare_param_setter(TYPE)                                             \
  {                                                                            \
    anari::ANARITypeFor<TYPE>::value,                                          \
        [](Object &o, const char *p, const void *v) {                          \
          o.setParam(p, *(TYPE *)v);                                           \
        }                                                                      \
  }

#define declare_param_setter_conversion(ENUM_TYPE_IN, CPP_TYPE_STORED)         \
  {                                                                            \
    ENUM_TYPE_IN, [](Object &o, const char *p, const void *v) {                \
      CPP_TYPE_STORED value;                                                   \
      std::memcpy(&value, v, sizeof(value));                                   \
      o.setParam(p, value);                                                    \
    }                                                                          \
  }

#define declare_param_setter_object(TYPE)                                      \
  {                                                                            \
    anari::ANARITypeFor<TYPE>::value,                                          \
        [](Object &o, const char *p, const void *v) {                          \
          using OBJECT_T = typename std::remove_pointer<TYPE>::type;           \
          auto ptr = *((TYPE *)v);                                             \
          if (ptr)                                                             \
            o.setParam(p, anari::IntrusivePtr<OBJECT_T>(ptr));                 \
          else                                                                 \
            o.removeParam(p);                                                  \
        }                                                                      \
  }

#define declare_param_setter_string(TYPE)                                      \
  {                                                                            \
    anari::ANARITypeFor<TYPE>::value,                                          \
        [](Object &o, const char *p, const void *v) {                          \
          const char *str = (const char *)v;                                   \
          o.setParam(p, std::string(str));                                     \
        }                                                                      \
  }

#define declare_param_setter_void_ptr(TYPE)                                    \
  {                                                                            \
    anari::ANARITypeFor<TYPE>::value,                                          \
        [](Object &o, const char *p, const void *v) {                          \
          o.setParam(p, const_cast<void *>(v));                                \
        }                                                                      \
  }

using SetParamFcn = void(Object &, const char *, const void *);

static std::map<ANARIDataType, SetParamFcn *> setParamFcns = {
    declare_param_setter(ANARIDataType),
    declare_param_setter(anari::Device *),
    declare_param_setter_void_ptr(void *),
    declare_param_setter(bool),
    declare_param_setter_object(Object *),
    declare_param_setter_object(Camera *),
    declare_param_setter_object(Array *),
    declare_param_setter_object(Array1D *),
    declare_param_setter_object(Array2D *),
    declare_param_setter_object(Array3D *),
    declare_param_setter_object(Frame *),
    declare_param_setter_object(Geometry *),
    declare_param_setter_object(Group *),
    declare_param_setter_object(Instance *),
    declare_param_setter_object(Light *),
    declare_param_setter_object(Material *),
    declare_param_setter_object(Renderer *),
    declare_param_setter_object(Sampler *),
    declare_param_setter_object(Surface *),
    declare_param_setter_object(SpatialField *),
    declare_param_setter_object(Volume *),
    declare_param_setter_object(World *),
    declare_param_setter_string(const char *),
    declare_param_setter_string(char *),
    declare_param_setter(int),
    declare_param_setter(unsigned int),
    declare_param_setter(float),
    declare_param_setter(ivec2),
    declare_param_setter(ivec3),
    declare_param_setter(ivec4),
    declare_param_setter(uvec2),
    declare_param_setter(uvec3),
    declare_param_setter(uvec4),
    declare_param_setter(vec2),
    declare_param_setter(vec3),
    declare_param_setter(vec4),
    declare_param_setter(mat4x3),
    declare_param_setter(mat4),
    declare_param_setter_conversion(ANARI_FLOAT32_BOX1, vec2),
    declare_param_setter_conversion(ANARI_FLOAT32_BOX2, vec4)};

#undef declare_param_setter
#undef declare_param_setter_object
#undef declare_param_setter_string
#undef declare_param_setter_void_ptr

///////////////////////////////////////////////////////////////////////////////
// VisRTXDevice definitions ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int VisRTXDevice::deviceImplements(const char *_extension)
{
  std::string_view extension = _extension;

  if (extension == "ANARI_KHR_STOCHASTIC_RENDERING")
    return 1;
  else if (extension == "ANARI_KHR_AUXILIARY_BUFFERS")
    return 1;
  else if (extension == "VISRTX_TRIANGLE_ATTRIBUTE_INDEXING")
    return 1;
  else if (extension == "VISRTX_SAMPLER_COLOR_MAP")
    return 1;
  else if (extension == "VISRTX_CUDA_OUTPUT_BUFFERS")
    return 1;

  return 0;
}

void VisRTXDevice::deviceSetParameter(
    const char *_id, ANARIDataType type, const void *mem)
{
  std::string id = _id;
  if (id == "statusCallback" && type == ANARI_STATUS_CALLBACK)
    setParam(id, (ANARIStatusCallback)mem);
  else if (id == "statusCallbackUserData" && type == ANARI_VOID_POINTER)
    setParam(id, mem);
  else if (id == "cudaDevice" && type == ANARI_INT32)
    setParam(id, *(int *)mem);
}

void VisRTXDevice::deviceUnsetParameter(const char *id)
{
  removeParam(id);
}

void VisRTXDevice::deviceCommit()
{
  m_statusCB =
      getParam<ANARIStatusCallback>("statusCallback", defaultStatusCallback());
  m_statusCBUserPtr = getParam<void *>(
      "statusCallbackUserData", defaultStatusCallbackUserPtr());
  m_desiredGpuID = getParam<int>("cudaDevice", 0);
  if (m_gpuID >= 0 && m_desiredGpuID != m_gpuID) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "visrtx was already initialized to use GPU %i"
        ": new device number %i is ignored.",
        m_gpuID,
        m_desiredGpuID);
  }
}

void VisRTXDevice::deviceRetain()
{
  this->refInc();
}

void VisRTXDevice::deviceRelease()
{
  this->refDec();
}

// Data Arrays ////////////////////////////////////////////////////////////////

ANARIArray1D VisRTXDevice::newArray1D(void *appMemory,
    ANARIMemoryDeleter deleter,
    void *userData,
    ANARIDataType type,
    uint64_t numItems,
    uint64_t byteStride)
{
  initDevice();
  CUDADeviceScope ds(this);
  if (anari::isObject(type)) {
    return createObjectForAPI<ObjectArray, ANARIArray1D>(m_state.get(),
        appMemory,
        deleter,
        userData,
        type,
        numItems,
        byteStride);
  } else {
    return createObjectForAPI<Array1D, ANARIArray1D>(m_state.get(),
        appMemory,
        deleter,
        userData,
        type,
        numItems,
        byteStride);
  }
}

ANARIArray2D VisRTXDevice::newArray2D(void *appMemory,
    ANARIMemoryDeleter deleter,
    void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t byteStride1,
    uint64_t byteStride2)
{
  initDevice();
  CUDADeviceScope ds(this);
  return createObjectForAPI<Array2D, ANARIArray2D>(m_state.get(),
      appMemory,
      deleter,
      userData,
      type,
      numItems1,
      numItems2,
      byteStride1,
      byteStride2);
}

ANARIArray3D VisRTXDevice::newArray3D(void *appMemory,
    ANARIMemoryDeleter deleter,
    void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3,
    uint64_t byteStride1,
    uint64_t byteStride2,
    uint64_t byteStride3)
{
  initDevice();
  CUDADeviceScope ds(this);
  return createObjectForAPI<Array3D, ANARIArray3D>(m_state.get(),
      appMemory,
      deleter,
      userData,
      type,
      numItems1,
      numItems2,
      numItems3,
      byteStride1,
      byteStride2,
      byteStride3);
}

void *VisRTXDevice::mapArray(ANARIArray a)
{
  CUDADeviceScope ds(this);
  return referenceFromHandle<Array>(a).map();
}

void VisRTXDevice::unmapArray(ANARIArray a)
{
  CUDADeviceScope ds(this);
  referenceFromHandle<Array>(a).unmap();
}

// Renderable Objects /////////////////////////////////////////////////////////

ANARILight VisRTXDevice::newLight(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return completeHandle<ANARILight>(
      Light::createInstance(subtype, m_state.get()));
}

ANARICamera VisRTXDevice::newCamera(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return completeHandle<ANARICamera>(
      Camera::createInstance(subtype, m_state.get()));
}

ANARIGeometry VisRTXDevice::newGeometry(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return completeHandle<ANARIGeometry>(
      Geometry::createInstance(subtype, m_state.get()));
}

ANARISpatialField VisRTXDevice::newSpatialField(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return completeHandle<ANARISpatialField>(
      SpatialField::createInstance(subtype, m_state.get()));
}

ANARISurface VisRTXDevice::newSurface()
{
  initDevice();
  CUDADeviceScope ds(this);
  return createObjectForAPI<Surface, ANARISurface>(m_state.get());
}

ANARIVolume VisRTXDevice::newVolume(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return completeHandle<ANARIVolume>(
      Volume::createInstance(subtype, m_state.get()));
}

// Surface Meta-Data //////////////////////////////////////////////////////////

ANARIMaterial VisRTXDevice::newMaterial(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return completeHandle<ANARIMaterial>(
      Material::createInstance(subtype, m_state.get()));
}

ANARISampler VisRTXDevice::newSampler(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return completeHandle<ANARISampler>(
      Sampler::createInstance(subtype, m_state.get()));
}

// Instancing /////////////////////////////////////////////////////////////////

ANARIGroup VisRTXDevice::newGroup()
{
  initDevice();
  CUDADeviceScope ds(this);
  return createObjectForAPI<Group, ANARIGroup>(m_state.get());
}

ANARIInstance VisRTXDevice::newInstance()
{
  initDevice();
  CUDADeviceScope ds(this);
  return createObjectForAPI<Instance, ANARIInstance>(m_state.get());
}

// Top-level Worlds ///////////////////////////////////////////////////////////

ANARIWorld VisRTXDevice::newWorld()
{
  initDevice();
  CUDADeviceScope ds(this);
  return createObjectForAPI<World, ANARIWorld>(m_state.get());
}

int VisRTXDevice::getProperty(ANARIObject object,
    const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    uint32_t mask)
{
  CUDADeviceScope ds(this);

  if (mask == ANARI_WAIT)
    m_state->flushCommitBuffer();

  if ((void *)object == (void *)this) {
    std::string_view prop = name;
    if (prop == "version" && type == ANARI_INT32) {
      int version = VISRTX_VERSION_MAJOR * 10000 + VISRTX_VERSION_MINOR * 100
          + VISRTX_VERSION_PATCH;
      writeToVoidP(mem, version);
      return 1;
    } else if (prop == "version.major" && type == ANARI_INT32) {
      writeToVoidP(mem, VISRTX_VERSION_MAJOR);
      return 1;
    } else if (prop == "version.minor" && type == ANARI_INT32) {
      writeToVoidP(mem, VISRTX_VERSION_MINOR);
      return 1;
    } else if (prop == "version.patch" && type == ANARI_INT32) {
      writeToVoidP(mem, VISRTX_VERSION_PATCH);
      return 1;
    }
  } else
    return referenceFromHandle(object).getProperty(name, type, mem, mask);

  return 0;
}

// Object + Parameter Lifetime Management /////////////////////////////////////

void VisRTXDevice::setParameter(
    ANARIObject object, const char *name, ANARIDataType type, const void *mem)
{
  if (type == ANARI_UNKNOWN)
    throw std::runtime_error("cannot set ANARI_UNKNOWN parameter type");

  auto *fcn = setParamFcns[type];
  auto &o = referenceFromHandle(object);

  if (fcn) {
    fcn(o, name, mem);
    o.markUpdated();
  } else {
    reportMessage(ANARI_SEVERITY_WARNING,
        "setting parameter type (%i) not yet implemented",
        type);
  }
}

void VisRTXDevice::unsetParameter(ANARIObject o, const char *name)
{
  auto &obj = referenceFromHandle(o);
  obj.removeParam(name);
  obj.markUpdated();
}

void VisRTXDevice::commit(ANARIObject o)
{
  m_state->commitBuffer.addObject((Object *)o);
}

void VisRTXDevice::release(ANARIObject o)
{
  if (o == nullptr)
    return;

  CUDADeviceScope ds(this);

  auto &obj = referenceFromHandle(o);

  if (obj.useCount(anari::RefType::PUBLIC) == 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected too many releases of object (type %i)",
        obj.type());
    return;
  }

  bool privatizeArray = anari::isArray(obj.type())
      && obj.useCount(anari::RefType::INTERNAL) > 0
      && obj.useCount(anari::RefType::PUBLIC) == 1;

  obj.refDec(anari::RefType::PUBLIC);

  if (privatizeArray)
    ((Array *)o)->privatize();
}

void VisRTXDevice::retain(ANARIObject o)
{
  referenceFromHandle(o).refInc(anari::RefType::PUBLIC);
}

// Frame Manipulation /////////////////////////////////////////////////////////

ANARIFrame VisRTXDevice::newFrame()
{
  initDevice();
  CUDADeviceScope ds(this);
  return createObjectForAPI<Frame, ANARIFrame>(m_state.get());
}

const void *VisRTXDevice::frameBufferMap(ANARIFrame fb, const char *channel)
{
  CUDADeviceScope ds(this);
  return referenceFromHandle<Frame>(fb).map(channel);
}

void VisRTXDevice::frameBufferUnmap(ANARIFrame, const char *)
{
  // no-op
}

// Frame Rendering ////////////////////////////////////////////////////////////

ANARIRenderer VisRTXDevice::newRenderer(const char *subtype)
{
  initDevice();
  CUDADeviceScope ds(this);
  return completeHandle<ANARIRenderer>(
      Renderer::createInstance(subtype, m_state.get()));
}

void VisRTXDevice::renderFrame(ANARIFrame frame)
{
  CUDADeviceScope ds(this);
  referenceFromHandle<Frame>(frame).renderFrame();
}

int VisRTXDevice::frameReady(ANARIFrame _f, ANARIWaitMask m)
{
  CUDADeviceScope ds(this);
  auto &f = referenceFromHandle<Frame>(_f);
  if (m == ANARI_NO_WAIT)
    return f.ready();
  else {
    f.wait();
    return 1;
  }
}

void VisRTXDevice::discardFrame(ANARIFrame)
{
  // no-op
}

// Other VisRTXDevice definitions /////////////////////////////////////////////

VisRTXDevice::~VisRTXDevice()
{
  if (m_state.get() == nullptr)
    return;

  auto &state = *m_state;

  state.commitBuffer.clear();
  state.uploadBuffer.clear();

  CUDA_SYNC_CHECK();

  optixModuleDestroy(state.rendererModules.debug);
  optixModuleDestroy(state.rendererModules.raycast);
  optixModuleDestroy(state.rendererModules.ambientOcclusion);
  optixModuleDestroy(state.rendererModules.diffusePathTracer);
  optixModuleDestroy(state.rendererModules.scivis);

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
  if (!state.registry.lights.empty()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked lights",
        state.registry.lights.size());
  }
  if (!state.registry.surfaces.empty()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked surfaces",
        state.registry.surfaces.size());
  }
  if (!state.registry.volumes.empty()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked volumes",
        state.registry.volumes.size());
  }
  if (!state.registry.geometries.empty()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked geometries",
        state.registry.geometries.size());
  }
  if (!state.registry.materials.empty()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked materials",
        state.registry.materials.size());
  }
  if (!state.registry.fields.empty()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked spatial fields",
        state.registry.fields.size());
  }
  if (!state.registry.samplers.empty()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked samplers",
        state.registry.samplers.size());
  }
  if (Array::objectCount() != 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked arrays",
        Array::objectCount());
  }
}

void VisRTXDevice::initDevice()
{
  if (m_state.get() != nullptr)
    return;

  deviceCommit();

  m_state = std::make_unique<DeviceGlobalState>();
  auto &state = *m_state;

  state.messageFunction = [&](ANARIStatusSeverity severity,
                              const std::string &msg,
                              const void *obj) {
    if (!m_statusCB)
      return;
    m_statusCB(m_statusCBUserPtr,
        this_device(),
        (ANARIObject)obj,
        ANARI_OBJECT,
        severity,
        severity >= ANARI_SEVERITY_WARNING ? ANARI_STATUS_NO_ERROR
                                           : ANARI_STATUS_UNKNOWN_ERROR,
        msg.c_str());
  };

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing VisRTX");

  cudaFree(nullptr);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0) {
    reportMessage(ANARI_SEVERITY_FATAL_ERROR, "no CUDA capable devices found!");
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

  auto init_module = [&](OptixModule &module, unsigned char *ptx) {
    const std::string ptxCode = (const char *)ptx;

    std::string log(2048, '\n');
    size_t sizeof_log = log.size();

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = PAYLOAD_VALUES;
    pipelineCompileOptions.numAttributeValues = ATTRIBUTE_VALUES;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "frameData";

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

  reportMessage(ANARI_SEVERITY_DEBUG, "Compiling custom intersectors");
  init_module(state.intersectionModules.customIntersectors, intersection_ptx());
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

VisRTXDevice::CUDADeviceScope::CUDADeviceScope(VisRTXDevice *d) : m_device(d)
{
  m_device->setCUDADevice();
}

VisRTXDevice::CUDADeviceScope::~CUDADeviceScope()
{
  m_device->revertCUDADevice();
}

} // namespace visrtx

#ifdef _WIN32
#ifdef VISRTX_DEVICE_STATIC_DEFINE
#define VISRTX_DEVICE_INTERFACE
#else
#ifdef anari_library_visrtx_EXPORTS
#define VISRTX_DEVICE_INTERFACE __declspec(dllexport)
#else
#define VISRTX_DEVICE_INTERFACE __declspec(dllimport)
#endif
#endif
#else
#define VISRTX_DEVICE_INTERFACE
#endif

extern "C" VISRTX_DEVICE_INTERFACE ANARI_DEFINE_LIBRARY_NEW_DEVICE(
    visrtx, _subtype)
{
  auto subtype = std::string_view(_subtype);
  if (subtype == "default" || subtype == "visrtx")
    return (ANARIDevice) new visrtx::VisRTXDevice();
  return nullptr;
}

extern "C" VISRTX_DEVICE_INTERFACE ANARI_DEFINE_LIBRARY_GET_DEVICE_SUBTYPES(
    visrtx, libdata)
{
  static const char *devices[] = {"visrtx", nullptr};
  return devices;
}

extern "C" VISRTX_DEVICE_INTERFACE ANARI_DEFINE_LIBRARY_GET_OBJECT_SUBTYPES(
    visrtx, libdata, deviceSubtype, objectType)
{
  if (objectType == ANARI_RENDERER) {
    static const char *renderers[] = {"scivis",
        "raycast",
        "ao",
        "diffuse_pathtracer",
        "debug_primID",
        "debug_geomID",
        "debug_instID",
        "debug_Ng",
        "debug_uvw",
        "debug_istri",
        "debug_isvol",
        "debug_backface",
        nullptr};
    return renderers;
  } else if (objectType == ANARI_GEOMETRY) {
    static const char *geometries[] = {
        "triangle", "quad", "cylinders", "sphere", nullptr};
    return geometries;
  } else if (objectType == ANARI_SPATIAL_FIELD) {
    static const char *spatialFields[] = {"structuredRegular", nullptr};
    return spatialFields;
  } else if (objectType == ANARI_VOLUME) {
    static const char *volumes[] = {"scivis", nullptr};
    return volumes;
  } else if (objectType == ANARI_MATERIAL) {
    static const char *materials[] = {
        "matte", "transparentMatte", "pbr", nullptr};
    return materials;
  } else if (objectType == ANARI_SAMPLER) {
    static const char *samplers[] = {
        "image2D", "primitive", "colorMap", nullptr};
    return samplers;
  } else if (objectType == ANARI_LIGHT) {
    static const char *lights[] = {"ambient", "directional", "point", nullptr};
    return lights;
  } else if (objectType == ANARI_CAMERA) {
    static const char *cameras[] = {"perspective", "orthographic", nullptr};
    return cameras;
  }

  return nullptr;
}

extern "C" VISRTX_DEVICE_INTERFACE ANARI_DEFINE_LIBRARY_GET_OBJECT_PARAMETERS(
    visrtx, libdata, deviceSubtype, objectSubtype, objectType)
{
  if (objectType == ANARI_RENDERER)
    return visrtx::Renderer::getParameters(objectSubtype);
  return nullptr;
}

extern "C" VISRTX_DEVICE_INTERFACE ANARI_DEFINE_LIBRARY_GET_PARAMETER_PROPERTY(
    visrtx,
    libdata,
    deviceSubtype,
    objectSubtype,
    objectType,
    parameterName,
    parameterType,
    infoName,
    infoType)
{
  if (objectType == ANARI_RENDERER) {
    return visrtx::Renderer::getParameterInfo(
        objectSubtype, parameterName, parameterType, infoName, infoType);
  }
  return nullptr;
}
