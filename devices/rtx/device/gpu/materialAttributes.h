// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "gpu_decl.h"

#include <cstdint>

namespace visrtx {

// Shared by host-side binding and GPU evaluation; no CUDA types are needed.
enum class MaterialAttribute : uint8_t
{
  ATTRIB_0,
  ATTRIB_1,
  ATTRIB_2,
  ATTRIB_3,
  COLOR,
  OBJECT_POSITION,
  OBJECT_NORMAL,
  WORLD_POSITION,
  WORLD_NORMAL,
  UNKNOWN
};

struct MDLSceneDataBinding
{
  MaterialAttribute attribute{MaterialAttribute::UNKNOWN};
  uint32_t nameToken{0};
};

// The material table is indexed by target-code-local string ID. Only the
// interned token can be compared with geometry aliases from another material.
VISRTX_HOST_DEVICE MaterialAttribute resolveSceneDataAttribute(
    const MDLSceneDataBinding &binding,
    const MDLSceneDataBinding *aliases,
    uint32_t numAliases)
{
  if (binding.nameToken != 0) {
    for (uint32_t i = 0; i < numAliases; ++i) {
      if (aliases[i].nameToken == binding.nameToken)
        return aliases[i].attribute;
    }
  }
  return binding.attribute;
}

} // namespace visrtx
