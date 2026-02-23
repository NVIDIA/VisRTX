// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anari/anari_cpp/Traits.h>
#include <anari/anari_cpp.hpp>

namespace tsd::core {

// TSD-specific type constants outside ANARI's reserved ranges
constexpr ANARIDataType TSD_TRANSFORM = ANARIDataType(10000);

inline bool isTSDTransform(ANARIDataType type)
{
  return type == TSD_TRANSFORM;
}

inline const char *tsdTypeName(ANARIDataType type)
{
  if (type == TSD_TRANSFORM)
    return "TSD_TRANSFORM";
  return "UNKNOWN_TSD_TYPE";
}

} // namespace tsd::core
