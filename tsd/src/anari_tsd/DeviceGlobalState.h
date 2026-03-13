// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_app
#include "tsd/app/Context.h"
// tsd_core
#include "tsd/core/ObjectVersion.hpp"
// helium
#include <helium/BaseGlobalDeviceState.h>
#include <helium/helium_math.h>

namespace tsd_device {

using TSDAny = tsd::core::Any;

struct DeviceGlobalState : public helium::BaseGlobalDeviceState
{
  DeviceGlobalState(anari::Device d);

  bool usingExternalScene() const;

  tsd::core::Scene *scene{nullptr};
  tsd::core::Scene localScene;
  tsd::app::ANARIDeviceManager anari;

  anari::Device device{nullptr};
  tsd::core::Token deviceName;

  struct ObjectUpdates
  {
    tsd::core::ObjectVersion instancing{};
  } objectUpdates;

  int cameraCount{0};
  int surfaceCount{0};
  int geometryCount{0};
  int materialCount{0};
  int samplerCount{0};
  int volumeCount{0};
  int fieldCount{0};
  int lightCount{0};
  int rendererCount{0};
  int worldCount{0};
};

// Helper functions/macros ////////////////////////////////////////////////////

inline DeviceGlobalState *asDeviceState(helium::BaseGlobalDeviceState *s)
{
  return (DeviceGlobalState *)s;
}

#define TSD_DEVICE_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)              \
  namespace anari {                                                            \
  ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);                              \
  }

#define TSD_DEVICE_ANARI_TYPEFOR_DEFINITION(type)                              \
  namespace anari {                                                            \
  ANARI_TYPEFOR_DEFINITION(type);                                              \
  }

} // namespace tsd_device
