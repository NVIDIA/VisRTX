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

// helium
#include "helium/BaseDevice.h"
// optix
#include "optix_visrtx.h"
// std
#include <atomic>

namespace visrtx {

enum class DeviceInitStatus
{
  UNINITIALIZED,
  SUCCESS,
  FAILURE
};

struct VisRTXDevice : public helium::BaseDevice
{
  /////////////////////////////////////////////////////////////////////////////
  // Main interface to accepting API calls
  /////////////////////////////////////////////////////////////////////////////

  void *mapArray(ANARIArray) override;
  void unmapArray(ANARIArray) override;

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

  ANARICamera newCamera(const char *type) override;
  ANARIFrame newFrame() override;
  ANARIGeometry newGeometry(const char *type) override;
  ANARIGroup newGroup() override;
  ANARIInstance newInstance(const char *type) override;
  ANARILight newLight(const char *type) override;
  ANARIMaterial newMaterial(const char *material_type) override;
  ANARIRenderer newRenderer(const char *type) override;
  ANARISampler newSampler(const char *type) override;
  ANARISpatialField newSpatialField(const char *type) override;
  ANARISurface newSurface() override;
  ANARIVolume newVolume(const char *type) override;
  ANARIWorld newWorld() override;

  const char **getObjectSubtypes(ANARIDataType objectType) override;
  const void *getObjectInfo(ANARIDataType objectType,
      const char *objectSubtype,
      const char *infoName,
      ANARIDataType infoType) override;
  const void *getParameterInfo(ANARIDataType objectType,
      const char *objectSubtype,
      const char *parameterName,
      ANARIDataType parameterType,
      const char *infoName,
      ANARIDataType infoType) override;

  int getProperty(ANARIObject object,
      const char *name,
      ANARIDataType type,
      void *mem,
      uint64_t size,
      uint32_t mask) override;

  const void *frameBufferMap(ANARIFrame fb,
      const char *channel,
      uint32_t *width,
      uint32_t *height,
      ANARIDataType *pixelType) override;

  void renderFrame(ANARIFrame) override;
  int frameReady(ANARIFrame, ANARIWaitMask) override;
  void discardFrame(ANARIFrame) override;

  /////////////////////////////////////////////////////////////////////////////
  // Helper/other functions and data members
  /////////////////////////////////////////////////////////////////////////////

  VisRTXDevice(ANARIStatusCallback defaultCallback, const void *userPtr);
  VisRTXDevice(ANARILibrary);
  ~VisRTXDevice() override;

  bool initDevice(); // thread safe initialization

 private:
  struct CUDADeviceScope
  {
    CUDADeviceScope(VisRTXDevice *d);
    ~CUDADeviceScope();

   private:
    VisRTXDevice *m_device{nullptr};
  };

  void deviceCommitParameters() override;
  int deviceGetProperty(const char *name,
      ANARIDataType type,
      void *mem,
      uint64_t size,
      uint32_t flags) override;

  DeviceInitStatus initOptix();
#ifdef USE_MDL
  DeviceInitStatus initMDL();
#endif // defined(USE_MDL)
  void setCUDADevice();
  void revertCUDADevice();

  DeviceGlobalState *deviceState() const;

  int m_gpuID{-1};
  int m_desiredGpuID{-1};
  int m_appGpuID{-1};
  bool m_eagerInit{false};
  std::atomic<DeviceInitStatus> m_initStatus{DeviceInitStatus::UNINITIALIZED};
#ifdef USE_MDL
  std::atomic<DeviceInitStatus> m_mdlInitStatus{
      DeviceInitStatus::UNINITIALIZED};
#endif // defined(USE_MDL)
};

} // namespace visrtx
