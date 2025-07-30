// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Device.h"

#include "Array.h"
#include "Frame.h"
#include "Object.h"

#include "anari_library_tsd_queries.h"

namespace tsd_device {

// API Objects ////////////////////////////////////////////////////////////////

ANARIArray1D Device::newArray1D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems)
{
  initDevice();

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

ANARIArray2D Device::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  initDevice();

  Array2DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;

  return (ANARIArray2D) new Array2D(deviceState(), md);
}

ANARIArray3D Device::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
  initDevice();

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

ANARICamera Device::newCamera(const char *subtype)
{
  initDevice();
  return (ANARICamera) new Object(ANARI_CAMERA, deviceState());
}

ANARIFrame Device::newFrame()
{
  initDevice();
  return (ANARIFrame) new Frame(deviceState());
}

ANARIGeometry Device::newGeometry(const char *subtype)
{
  initDevice();
  return (ANARIGeometry) new Object(ANARI_GEOMETRY, deviceState());
}

ANARIGroup Device::newGroup()
{
  initDevice();
  return (ANARIGroup) new Object(ANARI_GROUP, deviceState());
}

ANARIInstance Device::newInstance(const char * /*subtype*/)
{
  initDevice();
  return (ANARIInstance) new Object(ANARI_INSTANCE, deviceState());
}

ANARILight Device::newLight(const char *subtype)
{
  initDevice();
  return (ANARILight) new Object(ANARI_LIGHT, deviceState());
}

ANARIMaterial Device::newMaterial(const char *subtype)
{
  initDevice();
  return (ANARIMaterial) new Object(ANARI_MATERIAL, deviceState());
}

ANARIRenderer Device::newRenderer(const char *subtype)
{
  initDevice();
  return (ANARIRenderer) new Object(ANARI_RENDERER, deviceState());
}

ANARISampler Device::newSampler(const char *subtype)
{
  initDevice();
  return (ANARISampler) new Object(ANARI_SAMPLER, deviceState());
}

ANARISpatialField Device::newSpatialField(const char *subtype)
{
  initDevice();
  return (ANARISpatialField) new Object(ANARI_SPATIAL_FIELD, deviceState());
}

ANARISurface Device::newSurface()
{
  initDevice();
  return (ANARISurface) new Object(ANARI_SURFACE, deviceState());
}

ANARIVolume Device::newVolume(const char *subtype)
{
  initDevice();
  return (ANARIVolume) new Object(ANARI_VOLUME, deviceState());
}

ANARIWorld Device::newWorld()
{
  initDevice();
  return (ANARIWorld) new Object(ANARI_WORLD, deviceState());
}

// Query functions ////////////////////////////////////////////////////////////

const char **Device::getObjectSubtypes(ANARIDataType objectType)
{
  return tsd_device::query_object_types(objectType);
}

const void *Device::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  return tsd_device::query_object_info(
      objectType, objectSubtype, infoName, infoType);
}

const void *Device::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  return tsd_device::query_param_info(objectType,
      objectSubtype,
      parameterName,
      parameterType,
      infoName,
      infoType);
}

// Other Device definitions ///////////////////////////////////////////////////

Device::Device(ANARIStatusCallback cb, const void *ptr)
    : helium::BaseDevice(cb, ptr)
{
  m_state = std::make_unique<DeviceGlobalState>(this_device());
  deviceCommitParameters();
}

Device::Device(ANARILibrary l) : helium::BaseDevice(l)
{
  m_state = std::make_unique<DeviceGlobalState>(this_device());
  deviceCommitParameters();
}

Device::~Device()
{
  auto &state = *deviceState();
  state.commitBuffer.clear();
  reportMessage(ANARI_SEVERITY_DEBUG, "destroying TSD device (%p)", this);
}

void Device::initDevice()
{
  if (m_initialized)
    return;
  reportMessage(ANARI_SEVERITY_DEBUG, "initializing TSD device (%p)", this);
  m_initialized = true;
}

void Device::deviceCommitParameters()
{
  helium::BaseDevice::deviceCommitParameters();
}

int Device::deviceGetProperty(
    const char *name, ANARIDataType type, void *mem, uint64_t size)
{
  std::string_view prop = name;
  if (prop == "extension" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, query_extensions());
    return 1;
  } else if (prop == "tsd" && type == ANARI_BOOL) {
    helium::writeToVoidP(mem, true);
    return 1;
  }
  return 0;
}

DeviceGlobalState *Device::deviceState() const
{
  return (DeviceGlobalState *)helium::BaseDevice::m_state.get();
}

} // namespace tsd_device
