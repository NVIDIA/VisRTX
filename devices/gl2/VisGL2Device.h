// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// helium
#include "helium/BaseDevice.h"

#include "Object.h"

namespace visgl2 {

struct VisGL2Device : public helium::BaseDevice
{
  /////////////////////////////////////////////////////////////////////////////
  // Main interface to accepting API calls
  /////////////////////////////////////////////////////////////////////////////

  // API Objects //////////////////////////////////////////////////////////////

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

  // Query functions //////////////////////////////////////////////////////////

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

  /////////////////////////////////////////////////////////////////////////////
  // Helper/other functions and data members
  /////////////////////////////////////////////////////////////////////////////

  VisGL2Device(ANARIStatusCallback defaultCallback, const void *userPtr);
  VisGL2Device(ANARILibrary);
  ~VisGL2Device() override;

  void initDevice();

  void deviceCommitParameters() override;
  int deviceGetProperty(
      const char *name, ANARIDataType type, void *mem, uint64_t size) override;

 private:
  VisGL2DeviceGlobalState *deviceState() const;

  friend void anariDeviceReportStatus(ANARIDevice,
      ANARIStatusSeverity severity,
      ANARIStatusCode code,
      const char *format,
      ...);

  std::string m_glAPI{"OpenGL"};
  bool m_glDebug{false};
  void *m_eglDisplay{nullptr};
  bool m_initialized{false};
};

} // namespace visgl2
