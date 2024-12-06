// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd
#include "tsd/TSD.hpp"
// helium
#include <helium/BaseGlobalDeviceState.h>

namespace tsd_device {

using namespace tsd;
using TSDAny = tsd::utility::Any;

struct DeviceGlobalState : public helium::BaseGlobalDeviceState
{
  DeviceGlobalState(ANARIDevice d);

  tsd::Context ctx;
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
