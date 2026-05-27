// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../MathUsertype.hpp"
#include "RegisterVec.hpp"
#include "tsd/core/TSDMath.hpp"

#include <fmt/format.h>
#include <sol/sol.hpp>

namespace tsd::scripting {

void registerFloat4Usertype(sol::table &tsd)
{
  auto t = tsd.new_usertype<math::float4>("float4",
      sol::constructors<math::float4(),
          math::float4(float, float, float, float)>(),
      "x",
      &math::float4::x,
      "y",
      &math::float4::y,
      "z",
      &math::float4::z,
      "w",
      &math::float4::w,
      sol::meta_function::to_string,
      [](const math::float4 &v) {
        return fmt::format("float4({}, {}, {}, {})", v.x, v.y, v.z, v.w);
      });
  registerVecArithmetic(t);
}

} // namespace tsd::scripting
