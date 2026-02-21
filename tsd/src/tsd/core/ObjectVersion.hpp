// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <cstdint>

namespace tsd::core {

using ObjectVersion = uint64_t;

bool versionChanged(ObjectVersion &lastChecked, const ObjectVersion &current);

// Inlined definitions ////////////////////////////////////////////////////////

inline bool versionChanged(
    ObjectVersion &lastChecked, const ObjectVersion &current)
{
  if (lastChecked < current) {
    lastChecked = current;
    return true;
  }
  return false;
}

} // namespace tsd::core
