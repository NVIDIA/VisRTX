// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../MathUsertype.hpp"
#include "RegisterVec.hpp"
#include "tsd/core/TSDMath.hpp"

#include <fmt/format.h>
#include <sol/sol.hpp>

namespace tsd::scripting {

void registerFloat2Usertype(sol::table &tsd)
{
  auto t = tsd.new_usertype<math::float2>("float2",
      sol::constructors<math::float2(), math::float2(float, float)>(),
      "x",
      &math::float2::x,
      "y",
      &math::float2::y,
      sol::meta_function::to_string,
      [](const math::float2 &v) {
        return fmt::format("float2({}, {})", v.x, v.y);
      });
  registerVecArithmetic(t);
}

} // namespace tsd::scripting
