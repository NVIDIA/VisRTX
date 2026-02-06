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

#pragma once

// anari
#include "VisGLObject.h"
#include "VisGLObjects.h"
#include "VisGLDeviceObject.h"
#include "anari/backend/DeviceImpl.h"
#include "ConcurrentArray.h"

#include "ogl.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

// clang-format off
namespace visgl{

void anariRetainInternal(ANARIDevice, ANARIObject, ANARIObject);
void anariReleaseInternal(ANARIDevice, ANARIObject, ANARIObject);
void anariDeleteInternal(ANARIDevice, ANARIObject);
void anariReportStatus(ANARIDevice,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *format,
    ...);
// clang-format on

template <class T>
T deviceHandle(ANARIDevice d)
{
  anari::DeviceImpl *ad = reinterpret_cast<anari::DeviceImpl *>(d);
  return static_cast<T>(ad);
}

struct VisGLDevice : public anari::DeviceImpl
{
  // Data Arrays //////////////////////////////////////////////////////////////

  ANARIArray1D newArray1D(const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType,
      uint64_t numItems1) override;

  ANARIArray2D newArray2D(const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType,
      uint64_t numItems1,
      uint64_t numItems2) override;

  ANARIArray3D newArray3D(const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType,
      uint64_t numItems1,
      uint64_t numItems2,
      uint64_t numItems3) override;

  void *mapArray(ANARIArray) override;
  void unmapArray(ANARIArray) override;

  // Renderable Objects ///////////////////////////////////////////////////////

  ANARILight newLight(const char *type) override;

  ANARICamera newCamera(const char *type) override;

  ANARIGeometry newGeometry(const char *type) override;
  ANARISpatialField newSpatialField(const char *type) override;

  ANARISurface newSurface() override;
  ANARIVolume newVolume(const char *type) override;

  // Surface Meta-Data ////////////////////////////////////////////////////////

  ANARIMaterial newMaterial(const char *material_type) override;

  ANARISampler newSampler(const char *type) override;

  // Instancing ///////////////////////////////////////////////////////////////

  ANARIGroup newGroup() override;

  ANARIInstance newInstance(const char *type) override;

  // Top-level Worlds /////////////////////////////////////////////////////////

  ANARIWorld newWorld() override;

  // Query functions //////////////////////////////////////////////////////////

  const char ** getObjectSubtypes(ANARIDataType objectType) override;
  const void* getObjectInfo(ANARIDataType objectType,
      const char* objectSubtype,
      const char* infoName,
      ANARIDataType infoType) override;
  const void* getParameterInfo(ANARIDataType objectType,
      const char* objectSubtype,
      const char* parameterName,
      ANARIDataType parameterType,
      const char* infoName,
      ANARIDataType infoType) override;
  // Object + Parameter Lifetime Management ///////////////////////////////////

  int getProperty(ANARIObject object,
      const char *name,
      ANARIDataType type,
      void *mem,
      uint64_t size,
      ANARIWaitMask mask) override;

  void setParameter(ANARIObject object,
      const char *name,
      ANARIDataType type,
      const void *mem) override;

  void unsetParameter(ANARIObject object, const char *name) override;
  void unsetAllParameters(ANARIObject object) override;

  void* mapParameterArray1D(ANARIObject object,
      const char* name,
      ANARIDataType dataType,
      uint64_t numElements1,
      uint64_t *elementStride) override;
  void* mapParameterArray2D(ANARIObject object,
      const char* name,
      ANARIDataType dataType,
      uint64_t numElements1,
      uint64_t numElements2,
      uint64_t *elementStride) override;
  void* mapParameterArray3D(ANARIObject object,
      const char* name,
      ANARIDataType dataType,
      uint64_t numElements1,
      uint64_t numElements2,
      uint64_t numElements3,
      uint64_t *elementStride) override;
  void unmapParameterArray(ANARIObject object,
      const char* name) override;

  void commitParameters(ANARIObject object) override;

  void release(ANARIObject) override;
  void retain(ANARIObject) override;
  void retainInternal(ANARIObject, ANARIObject);
  void releaseInternal(ANARIObject, ANARIObject);

  // FrameBuffer Manipulation /////////////////////////////////////////////////

  ANARIFrame newFrame() override;

  const void *frameBufferMap(ANARIFrame fb,
      const char *channel,
      uint32_t *width,
      uint32_t *height,
      ANARIDataType *pixelType) override;

  void frameBufferUnmap(ANARIFrame fb, const char *channel) override;

  // Frame Rendering //////////////////////////////////////////////////////////

  ANARIRenderer newRenderer(const char *type) override;

  void renderFrame(ANARIFrame) override;
  int frameReady(ANARIFrame, ANARIWaitMask) override;
  void discardFrame(ANARIFrame) override;

  /////////////////////////////////////////////////////////////////////////////
  // Helper/other functions and data members
  /////////////////////////////////////////////////////////////////////////////

  VisGLDevice(ANARILibrary);
  ~VisGLDevice();

  ObjectBase *fromHandle(ANARIObject handle);
  template <class T, class H>
  T handle_cast(H handle)
  {
    ObjectBase *base = fromHandle(handle);
    if (base
        && is_convertible<typename std::remove_pointer<T>::type>::check(base)) {
      return static_cast<T>(base);
    } else {
      return nullptr;
    }
  }

 private:
  // object allocation and translation

  friend void anariDeleteInternal(ANARIDevice, ANARIObject);
  friend void anariReportStatus(ANARIDevice,
      ANARIObject source,
      ANARIDataType sourceType,
      ANARIStatusSeverity severity,
      ANARIStatusCode code,
      const char *format,
      ...);

  template <typename R, typename T, typename... ARGS>
  R allocate(ARGS... args)
  {
    uintptr_t idx = objects.add();
    ANARIObject handle = reinterpret_cast<ANARIObject>(idx);
    if (handle == this_device()) {
      idx = objects.add();
      handle = reinterpret_cast<ANARIObject>(idx);
    }
    objects[idx].reset(
        ObjectAllocator<T>::allocate(this_device(), handle, args...));
    objects[idx]->init();
    return static_cast<R>(handle);
  }

  void deallocate(ANARIObject handle)
  {
    uintptr_t id = reinterpret_cast<uintptr_t>(handle);
    if (id < objects.size()) {
      return objects[id].reset(nullptr);
    }
  }

  ANARIStatusCallback statusCallback;
  const void *statusCallbackUserData;

  std::atomic<int64_t> refcount;
  std::recursive_mutex mutex;
  Object<visgl::Device> deviceObject;
  ConcurrentArray<std::unique_ptr<ObjectBase>> objects;
};

template <class T, class H>
T handle_cast(ANARIDevice d, H handle)
{
  return deviceHandle<VisGLDevice *>(d)->handle_cast<T>(handle);
}

template <class T>
T ObjectBase::handle_cast(ANARIObject h)
{
  return visgl::handle_cast<T>(device, h);
}

template <class T>
T ObjectBase::acquire(ANARIObject h)
{
  T obj = visgl::handle_cast<T>(device, h);
  if (obj) {
    obj->update();
  }
  return obj;
}

template <class T>
T ObjectBase::handle_cast(ParameterBase &h)
{
  return handle_cast<T>(h.getHandle());
}

template <class T>
T ObjectBase::acquire(ParameterBase &h)
{
  return acquire<T>(h.getHandle());
}

} // namespace visgl
