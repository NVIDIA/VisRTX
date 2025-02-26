// Copyright 2021-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "VisGL2Device.h"

#include "array/Array1D.h"
#include "array/Array2D.h"
#include "array/Array3D.h"
#include "array/ObjectArray.h"
#include "frame/Frame.h"
#include "scene/volume/spatial_field/SpatialField.h"

#include "anari_library_visgl2_queries.h"

namespace visgl2 {

// API Objects ////////////////////////////////////////////////////////////////

ANARIArray1D VisGL2Device::newArray1D(const void *appMemory,
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

ANARIArray2D VisGL2Device::newArray2D(const void *appMemory,
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

ANARIArray3D VisGL2Device::newArray3D(const void *appMemory,
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

ANARICamera VisGL2Device::newCamera(const char *subtype)
{
  initDevice();
  return (ANARICamera)Camera::createInstance(subtype, deviceState());
}

ANARIFrame VisGL2Device::newFrame()
{
  initDevice();
  return (ANARIFrame) new Frame(deviceState());
}

ANARIGeometry VisGL2Device::newGeometry(const char *subtype)
{
  initDevice();
  return (ANARIGeometry)Geometry::createInstance(subtype, deviceState());
}

ANARIGroup VisGL2Device::newGroup()
{
  initDevice();
  return (ANARIGroup) new Group(deviceState());
}

ANARIInstance VisGL2Device::newInstance(const char * /*subtype*/)
{
  initDevice();
  return (ANARIInstance) new Instance(deviceState());
}

ANARILight VisGL2Device::newLight(const char *subtype)
{
  initDevice();
  return (ANARILight)Light::createInstance(subtype, deviceState());
}

ANARIMaterial VisGL2Device::newMaterial(const char *subtype)
{
  initDevice();
  return (ANARIMaterial)Material::createInstance(subtype, deviceState());
}

ANARIRenderer VisGL2Device::newRenderer(const char *subtype)
{
  initDevice();
  return (ANARIRenderer)Renderer::createInstance(subtype, deviceState());
}

ANARISampler VisGL2Device::newSampler(const char *subtype)
{
  initDevice();
  return (ANARISampler)Sampler::createInstance(subtype, deviceState());
}

ANARISpatialField VisGL2Device::newSpatialField(const char *subtype)
{
  initDevice();
  return (ANARISpatialField)SpatialField::createInstance(
      subtype, deviceState());
}

ANARISurface VisGL2Device::newSurface()
{
  initDevice();
  return (ANARISurface) new Surface(deviceState());
}

ANARIVolume VisGL2Device::newVolume(const char *subtype)
{
  initDevice();
  return (ANARIVolume)Volume::createInstance(subtype, deviceState());
}

ANARIWorld VisGL2Device::newWorld()
{
  initDevice();
  return (ANARIWorld) new World(deviceState());
}

// Query functions ////////////////////////////////////////////////////////////

const char **VisGL2Device::getObjectSubtypes(ANARIDataType objectType)
{
  return visgl2::query_object_types(objectType);
}

const void *VisGL2Device::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  return visgl2::query_object_info(
      objectType, objectSubtype, infoName, infoType);
}

const void *VisGL2Device::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  return visgl2::query_param_info(objectType,
      objectSubtype,
      parameterName,
      parameterType,
      infoName,
      infoType);
}

// Other VisGL2Device definitions /////////////////////////////////////////////

VisGL2Device::VisGL2Device(ANARIStatusCallback cb, const void *ptr)
    : helium::BaseDevice(cb, ptr)
{
  m_state = std::make_unique<VisGL2DeviceGlobalState>(this_device());
  deviceCommitParameters();
}

VisGL2Device::VisGL2Device(ANARILibrary l) : helium::BaseDevice(l)
{
  m_state = std::make_unique<VisGL2DeviceGlobalState>(this_device());
  deviceCommitParameters();
}

VisGL2Device::~VisGL2Device()
{
  auto &state = *deviceState();
  state.commitBuffer.clear();
  reportMessage(ANARI_SEVERITY_DEBUG, "destroying VisGL2 device (%p)", this);
}

void VisGL2Device::initDevice()
{
  if (m_initialized)
    return;
  reportMessage(ANARI_SEVERITY_DEBUG, "initializing VisGL2 device (%p)", this);
  m_initialized = true;
}

void VisGL2Device::deviceCommitParameters()
{
  helium::BaseDevice::deviceCommitParameters();
}

int VisGL2Device::deviceGetProperty(
    const char *name, ANARIDataType type, void *mem, uint64_t size)
{
  std::string_view prop = name;
  if (prop == "extension" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, query_extensions());
    return 1;
  } else if (prop == "VisGL2" && type == ANARI_BOOL) {
    helium::writeToVoidP(mem, true);
    return 1;
  }
  return 0;
}

VisGL2DeviceGlobalState *VisGL2Device::deviceState() const
{
  return (VisGL2DeviceGlobalState *)helium::BaseDevice::m_state.get();
}

} // namespace visgl2
