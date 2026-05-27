// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../MathUsertype.hpp"
#include "tsd/core/TSDMath.hpp"

#include <fmt/format.h>
#include <sol/sol.hpp>

#include <stdexcept>

namespace tsd::scripting {

void registerMat4Usertype(sol::table &tsd)
{
  tsd.new_usertype<math::mat4>("mat4",
      sol::constructors<math::mat4(),
          math::mat4(
              math::float4, math::float4, math::float4, math::float4)>(),
      sol::meta_function::multiplication,
      sol::overload([](const math::mat4 &a,
                        const math::mat4 &b) { return math::mul(a, b); },
          [](const math::mat4 &a, const math::float4 &v) {
            return math::mul(a, v);
          }),
      sol::meta_function::index,
      [](sol::this_state L, const math::mat4 &m, sol::object key) -> sol::object {
        if (key.is<int>()) {
          int i = key.as<int>();
          if (i < 0 || i > 3)
            throw std::out_of_range("mat4 index must be 0, 1, 2, or 3");
          return sol::make_object(L, m[i]);
        }
        if (key.is<std::string_view>() && key.as<std::string_view>() == "identity")
          return sol::make_object(L, math::IDENTITY_MAT4);
        return sol::lua_nil;
      },
      sol::meta_function::new_index,
      [](math::mat4 &m, int i, const math::float4 &v) {
        if (i < 0 || i > 3)
          throw std::out_of_range("mat4 index must be 0, 1, 2, or 3");
        m[i] = v;
      },
      sol::meta_function::to_string,
      [](const math::mat4 &m) {
        return fmt::format("mat4({}, {}, {}, {})",
            fmt::format(
                "float4({}, {}, {}, {})", m[0].x, m[0].y, m[0].z, m[0].w),
            fmt::format(
                "float4({}, {}, {}, {})", m[1].x, m[1].y, m[1].z, m[1].w),
            fmt::format(
                "float4({}, {}, {}, {})", m[2].x, m[2].y, m[2].z, m[2].w),
            fmt::format(
                "float4({}, {}, {}, {})", m[3].x, m[3].y, m[3].z, m[3].w));
      });
}

} // namespace tsd::scripting
