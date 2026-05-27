// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../MathUsertype.hpp"
#include "tsd/core/TSDMath.hpp"

#include <fmt/format.h>
#include <sol/sol.hpp>

#include <stdexcept>

namespace tsd::scripting {

void registerMat3Usertype(sol::table &tsd)
{
  tsd.new_usertype<math::mat3>(
      "mat3",
      sol::constructors<math::mat3(),
          math::mat3(math::float3, math::float3, math::float3)>(),
      sol::meta_function::index,
      [](sol::this_state L, const math::mat3 &m, sol::object key) -> sol::object {
        if (key.is<int>()) {
          int i = key.as<int>();
          if (i < 0 || i > 2)
            throw std::out_of_range("mat3 index must be 0, 1, or 2");
          return sol::make_object(L, m[i]);
        }
        if (key.is<std::string_view>() && key.as<std::string_view>() == "identity")
          return sol::make_object(L, math::IDENTITY_MAT3);
        return sol::lua_nil;
      },
      sol::meta_function::new_index,
      [](math::mat3 &m, int i, const math::float3 &v) {
        if (i < 0 || i > 2)
          throw std::out_of_range("mat3 index must be 0, 1, or 2");
        m[i] = v;
      },
      sol::meta_function::to_string,
      [](const math::mat3 &m) {
        return fmt::format("mat3({}, {}, {})",
            fmt::format("float3({}, {}, {})", m[0].x, m[0].y, m[0].z),
            fmt::format("float3({}, {}, {})", m[1].x, m[1].y, m[1].z),
            fmt::format("float3({}, {}, {})", m[2].x, m[2].y, m[2].z));
      });
}

} // namespace tsd::scripting
