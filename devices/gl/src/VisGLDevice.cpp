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

#include <cstdarg>
#include <cstdint>
#include "VisGLDevice.h"

namespace visgl {

template <typename T>
void writeToVoidP(void *_p, T v)
{
  T *p = (T *)_p;
  *p = v;
}

void *VisGLDevice::mapArray(ANARIArray handle)
{
  if (auto obj = handle_cast<ArrayObjectBase *>(handle)) {
    return obj->map();
  } else {
    return nullptr;
  }
}
void VisGLDevice::unmapArray(ANARIArray handle)
{
  if (auto obj = handle_cast<ArrayObjectBase *>(handle)) {
    obj->unmap();
  }
}

const char ** query_extensions();
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

const char ** VisGLDevice::getObjectSubtypes(ANARIDataType objectType)
{
  return query_object_types(objectType);
}

const void* VisGLDevice::getObjectInfo(ANARIDataType objectType,
    const char* objectSubtype,
    const char* infoName,
    ANARIDataType infoType)
{
  return query_object_info(
      objectType, objectSubtype, infoName, infoType);
}

const void* VisGLDevice::getParameterInfo(ANARIDataType objectType,
    const char* objectSubtype,
    const char* parameterName,
    ANARIDataType parameterType,
    const char* infoName,
    ANARIDataType infoType)
{
  return query_param_info(objectType,
      objectSubtype,
      parameterName,
      parameterType,
      infoName,
      infoType);
}

int VisGLDevice::getProperty(ANARIObject handle,
    const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    ANARIWaitMask mask)
{
  if (auto obj = handle_cast<ObjectBase *>(handle)) {
    return obj->getProperty(name, type, mem, size, mask);
  } else {
    return 0;
  }
}

void VisGLDevice::setParameter(
    ANARIObject handle, const char *name, ANARIDataType type, const void *mem)
{
  if (auto obj = handle_cast<ObjectBase *>(handle)) {
    obj->set(name, type, mem);
  }
}

void VisGLDevice::unsetParameter(ANARIObject handle, const char *name)
{
  if (auto obj = handle_cast<ObjectBase *>(handle)) {
    obj->unset(name);
  }
}

void VisGLDevice::unsetAllParameters(ANARIObject handle)
{
  if (auto obj = handle_cast<ObjectBase *>(handle)) {
    obj->unsetAll();
  }
}

void* VisGLDevice::mapParameterArray1D(ANARIObject handle,
    const char* name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t* elementStride)
{
  if (auto obj = handle_cast<ObjectBase *>(handle)) {
    return obj->mapParameter1D(name,
      dataType,
      numElements1,
      elementStride);
  } else {
    return nullptr;
  }
}

void* VisGLDevice::mapParameterArray2D(ANARIObject handle,
    const char* name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t* elementStride)
{
  if (auto obj = handle_cast<ObjectBase *>(handle)) {
    return obj->mapParameter2D(name,
      dataType,
      numElements1,
      numElements2,
      elementStride);
  } else {
    return nullptr;
  }
}

void* VisGLDevice::mapParameterArray3D(ANARIObject handle,
    const char* name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t numElements3,
    uint64_t* elementStride)
{
  if (auto obj = handle_cast<ObjectBase *>(handle)) {
    return obj->mapParameter3D(name,
      dataType,
      numElements1,
      numElements2,
      numElements3,
      elementStride);
  } else {
    return nullptr;
  }
}

void VisGLDevice::unmapParameterArray(ANARIObject handle,
    const char* name)
{
  if (auto obj = handle_cast<ObjectBase *>(handle)) {
    return obj->unmapParameter(name);
  }
}

void VisGLDevice::commitParameters(ANARIObject handle)
{
  if (auto obj = handle_cast<ObjectBase *>(handle)) {
    obj->commit();
  }
  if (handle == this_device()) {
    if (deviceObject.current.statusCallback.get(
            ANARI_STATUS_CALLBACK, &statusCallback)) {
      statusCallbackUserData = nullptr;
      deviceObject.current.statusCallbackUserData.get(
          ANARI_VOID_POINTER, &statusCallbackUserData);
    } else {
      statusCallback = defaultStatusCallback();
      statusCallbackUserData = defaultStatusCallbackUserPtr();
    }
  }
}

void VisGLDevice::release(ANARIObject handle)
{
  if (handle == this_device()) {
    if (refcount.fetch_sub(1) == 1) {
      delete this;
    }
  } else if (auto obj = handle_cast<ObjectBase *>(handle)) {
    obj->release();
  }
}

void VisGLDevice::retain(ANARIObject handle)
{
  if (handle == this_device()) {
    refcount++;
  } else if (auto obj = handle_cast<ObjectBase *>(handle)) {
    obj->retain();
  }
}

void VisGLDevice::releaseInternal(ANARIObject handle, ANARIObject owner)
{
  if (auto obj = handle_cast<ObjectBase *>(handle)) {
    obj->releaseInternal(owner);
  }
}

void VisGLDevice::retainInternal(ANARIObject handle, ANARIObject owner)
{
  if (auto obj = handle_cast<ObjectBase *>(handle)) {
    obj->retainInternal(owner);
  }
}

const void *VisGLDevice::frameBufferMap(ANARIFrame handle,
    const char *channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  if (auto obj = handle_cast<FrameObjectBase *>(handle)) {
    return obj->mapFrame(channel, width, height, pixelType);
  } else {
    return 0;
  }
}

void VisGLDevice::frameBufferUnmap(ANARIFrame handle, const char *channel)
{
  if (auto obj = handle_cast<FrameObjectBase *>(handle)) {
    obj->unmapFrame(channel);
  }
}

void VisGLDevice::renderFrame(ANARIFrame handle)
{
  if (auto obj = handle_cast<FrameObjectBase *>(handle)) {
    obj->renderFrame();
  }
}
int VisGLDevice::frameReady(ANARIFrame handle, ANARIWaitMask mask)
{
  if (auto obj = handle_cast<FrameObjectBase *>(handle)) {
    return obj->frameReady(mask);
  } else {
    return 0;
  }
}
void VisGLDevice::discardFrame(ANARIFrame handle)
{
  if (auto obj = handle_cast<FrameObjectBase *>(handle)) {
    obj->discardFrame();
  }
}

/////////////////////////////////////////////////////////////////////////////
// Helper/other functions and data members
/////////////////////////////////////////////////////////////////////////////

VisGLDevice::VisGLDevice(ANARILibrary library)
    : DeviceImpl(library), refcount(1), deviceObject(this_device())
{
  objects.add(); // reserve the null index for the null handle
  statusCallback = defaultStatusCallback();
  statusCallbackUserData = defaultStatusCallbackUserPtr();
}

VisGLDevice::~VisGLDevice()
{
  for (uint64_t i = 0; i < objects.size(); ++i) {
    objects[i].reset(nullptr);
  }
}

ObjectBase *VisGLDevice::fromHandle(ANARIObject handle)
{
  if (handle == static_cast<ANARIObject>(this_device())) {
    return &deviceObject;
  }

  uintptr_t idx = reinterpret_cast<uintptr_t>(handle);
  if (idx < objects.size()) {
    return objects[idx].get();
  } else {
    return nullptr;
  }
}

// query functions
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

// internal "api" functions
void anariRetainInternal(ANARIDevice d, ANARIObject handle, ANARIObject owner)
{
  reinterpret_cast<VisGLDevice *>(d)->retainInternal(handle, owner);
}
void anariReleaseInternal(ANARIDevice d, ANARIObject handle, ANARIObject owner)
{
  reinterpret_cast<VisGLDevice *>(d)->releaseInternal(handle, owner);
}
void anariDeleteInternal(ANARIDevice d, ANARIObject handle)
{
  reinterpret_cast<VisGLDevice *>(d)->deallocate(handle);
}
void anariReportStatus(ANARIDevice handle,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *format,
    ...)
{
  if (VisGLDevice *d = deviceHandle<VisGLDevice *>(handle)) {
    if (d->statusCallback) {
      va_list arglist;
      va_list arglist_copy;
      va_start(arglist, format);
      va_copy(arglist_copy, arglist);
      int count = std::vsnprintf(nullptr, 0, format, arglist);
      va_end(arglist);

      std::vector<char> formattedMessage(size_t(count + 1));

      std::vsnprintf(
          formattedMessage.data(), size_t(count + 1), format, arglist_copy);
      va_end(arglist_copy);

      d->statusCallback(d->statusCallbackUserData,
          d->this_device(),
          source,
          sourceType,
          severity,
          code,
          formattedMessage.data());
    }
  }
}

} // namespace visgl
