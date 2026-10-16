// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

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
};

} // namespace visrtx
