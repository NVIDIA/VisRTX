// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/anari_cpp.hpp>
// helium
#include <helium/helium_math.h>

namespace tsd {
namespace math {

using namespace anari::math;
using namespace helium::math;

static constexpr mat4 IDENTITY_MAT4 = identity;

inline float radians(float degrees)
{
  return degrees * M_PI / 180.f;
};

} // namespace math

using namespace linalg::aliases;
using mat4 = float4x4;

} // namespace tsd
