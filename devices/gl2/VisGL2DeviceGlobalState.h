// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// helium
#include "helium/BaseGlobalDeviceState.h"
// std
#include <memory>

#include "GLContextInterface.h"
#include "TaskQueue.h"
#include "ogl.h"

namespace visgl2 {

struct VisGL2DeviceGlobalState : public helium::BaseGlobalDeviceState
{
  struct GLContextState
  {
    tasking::TaskQueue thread{128};
    std::unique_ptr<GLContextInterface> context;
    bool useGLES{false};
    GladGLContext glAPI{};
    std::vector<const char *> extensions;
  } gl;

  ANARIDevice device{nullptr};

  VisGL2DeviceGlobalState(ANARIDevice d);
};

// Helper functions/macros ////////////////////////////////////////////////////

#define VISGL2_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);                              \
  }

#define VISGL2_ANARI_TYPEFOR_DEFINITION(type)                                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_DEFINITION(type);                                              \
  }

} // namespace visgl2
