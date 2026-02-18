// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/TSDMath.hpp"
#include "tsd/scripting/LuaBindings.hpp"

#include <fmt/format.h>
#include <sol/sol.hpp>

namespace tsd::scripting {

template <typename T>
static void registerVecArithmetic(sol::usertype<T> &ut)
{
  ut[sol::meta_function::addition] = [](const T &a, const T &b) {
    return a + b;
  };
  ut[sol::meta_function::subtraction] = [](const T &a, const T &b) {
    return a - b;
  };
  ut[sol::meta_function::multiplication] =
      [](sol::object lhs, sol::object rhs) -> T {
    if (lhs.is<T>() && rhs.is<T>())
      return lhs.as<T>() * rhs.as<T>();
    if (lhs.is<T>() && rhs.is<double>())
      return lhs.as<T>() * static_cast<float>(rhs.as<double>());
    if (lhs.is<double>() && rhs.is<T>())
      return static_cast<float>(lhs.as<double>()) * rhs.as<T>();
    throw std::runtime_error("invalid operand types for *");
  };
  ut[sol::meta_function::division] =
      [](sol::object lhs, sol::object rhs) -> T {
    if (lhs.is<T>() && rhs.is<T>())
      return lhs.as<T>() / rhs.as<T>();
    if (lhs.is<T>() && rhs.is<double>())
      return lhs.as<T>() / static_cast<float>(rhs.as<double>());
    throw std::runtime_error("invalid operand types for /");
  };
  ut[sol::meta_function::unary_minus] = [](const T &a) { return -a; };
}

void registerMathBindings(sol::state &lua)
{
  sol::table tsd = lua["tsd"];

  auto float2Type = tsd.new_usertype<math::float2>("float2",
      sol::constructors<math::float2(), math::float2(float, float)>(),
      "x",
      &math::float2::x,
      "y",
      &math::float2::y,
      sol::meta_function::to_string,
      [](const math::float2 &v) {
        return fmt::format("float2({}, {})", v.x, v.y);
      });
  registerVecArithmetic(float2Type);

  auto float3Type = tsd.new_usertype<math::float3>("float3",
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
  registerVecArithmetic(float3Type);

  auto float4Type = tsd.new_usertype<math::float4>("float4",
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
  registerVecArithmetic(float4Type);

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

  tsd["float2"] =
      sol::overload([]() { return math::float2(); },
          [](float x, float y) { return math::float2(x, y); });

  tsd["float3"] =
      sol::overload([]() { return math::float3(); },
          [](float x, float y, float z) { return math::float3(x, y, z); });

  tsd["float4"] =
      sol::overload([]() { return math::float4(); },
          [](float x, float y, float z, float w) {
            return math::float4(x, y, z, w);
          });

  auto mat3Call = sol::overload(
      []() { return math::mat3(); },
      [](const math::float3 &col0,
          const math::float3 &col1,
          const math::float3 &col2) { return math::mat3(col0, col1, col2); });
  sol::table mat3Ctor = lua.create_table();
  sol::table mat3CtorMt = lua.create_table();
  mat3Ctor["new"] = mat3Call;
  mat3CtorMt[sol::meta_function::call] = sol::overload(
      [](sol::table) { return math::mat3(); },
      [](sol::table,
          const math::float3 &col0,
          const math::float3 &col1,
          const math::float3 &col2) { return math::mat3(col0, col1, col2); });
  mat3Ctor[sol::metatable_key] = mat3CtorMt;
  tsd["mat3"] = mat3Ctor;

  auto mat4Call = sol::overload(
      []() { return math::mat4(); },
      [](const math::float4 &col0,
          const math::float4 &col1,
          const math::float4 &col2,
          const math::float4 &col3) {
        return math::mat4(col0, col1, col2, col3);
      });
  sol::table mat4Ctor = lua.create_table();
  sol::table mat4CtorMt = lua.create_table();
  mat4Ctor["new"] = mat4Call;
  mat4CtorMt[sol::meta_function::call] = sol::overload(
      [](sol::table) { return math::mat4(); },
      [](sol::table,
          const math::float4 &col0,
          const math::float4 &col1,
          const math::float4 &col2,
          const math::float4 &col3) {
        return math::mat4(col0, col1, col2, col3);
      });
  mat4Ctor[sol::metatable_key] = mat4CtorMt;
  tsd["mat4"] = mat4Ctor;

  tsd["mat3"]["identity"] = math::IDENTITY_MAT3;
  tsd["mat4"]["identity"] = math::IDENTITY_MAT4;
  tsd["srt"] = tsd["mat3"];

  tsd["length"] =
      sol::overload([](const math::float2 &v) { return math::length(v); },
          [](const math::float3 &v) { return math::length(v); },
          [](const math::float4 &v) { return math::length(v); });

  tsd["normalize"] =
      sol::overload([](const math::float2 &v) { return math::normalize(v); },
          [](const math::float3 &v) { return math::normalize(v); },
          [](const math::float4 &v) { return math::normalize(v); });

  tsd["dot"] =
      sol::overload([](const math::float2 &a,
                        const math::float2 &b) { return math::dot(a, b); },
          [](const math::float3 &a, const math::float3 &b) {
            return math::dot(a, b);
          },
          [](const math::float4 &a, const math::float4 &b) {
            return math::dot(a, b);
          });

  tsd["cross"] = [](const math::float3 &a, const math::float3 &b) {
    return math::cross(a, b);
  };

  // Transform matrices
  tsd["translation"] = [](const math::float3 &t) {
    return math::translation_matrix(t);
  };

  tsd["scaling"] = sol::overload(
      [](const math::float3 &s) { return math::scaling_matrix(s); },
      [](float s) { return math::scaling_matrix(math::float3(s, s, s)); });

  tsd["rotation"] = [](const math::float3 &axis, float angle) {
    return math::rotation_matrix(math::rotation_quat(axis, angle));
  };

  tsd["radians"] = [](float degrees) { return math::radians(degrees); };
  tsd["degrees"] = [](float radians) { return math::degrees(radians); };
}

} // namespace tsd::scripting
