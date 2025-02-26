// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// helium
#include "helium/BaseGlobalDeviceState.h"

namespace visgl2 {

struct VisGL2DeviceGlobalState : public helium::BaseGlobalDeviceState
{
  // Add any data members here which all objects need to be able to access

  VisGL2DeviceGlobalState(ANARIDevice d);
};

// Helper functions/macros ////////////////////////////////////////////////////

inline VisGL2DeviceGlobalState *asVisGL2DeviceState(
    helium::BaseGlobalDeviceState *s)
{
  return (VisGL2DeviceGlobalState *)s;
}

#define VISGL2_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);                              \
  }

#define VISGL2_ANARI_TYPEFOR_DEFINITION(type)                                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_DEFINITION(type);                                              \
  }

} // namespace visgl2
