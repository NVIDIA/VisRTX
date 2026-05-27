// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../MathUsertype.hpp"
#include "RegisterVec.hpp"
#include "tsd/core/TSDMath.hpp"

#include <fmt/format.h>
#include <sol/sol.hpp>

namespace tsd::scripting {

void registerFloat3Usertype(sol::table &tsd)
{
  auto t = tsd.new_usertype<math::float3>("float3",
      sol::constructors<math::float3(), math::float3(float, float, float)>(),
      "x",
      &math::float3::x,
      "y",
      &math::float3::y,
      "z",
      &math::float3::z,
      sol::meta_function::to_string,
      [](const math::float3 &v) {
        return fmt::format("float3({}, {}, {})", v.x, v.y, v.z);
      });
  registerVecArithmetic(t);
}

} // namespace tsd::scripting
