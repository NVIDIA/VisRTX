// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "MathUsertype.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/scripting/LuaBindings.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerMathBindings(sol::state &lua)
{
  sol::table tsd = lua["tsd"];

  // Each math usertype lives in its own TU under bindings/math/ so the
  // five sol2 instantiations parallelise across cores.
  registerFloat2Usertype(tsd);
  registerFloat3Usertype(tsd);
  registerFloat4Usertype(tsd);
  registerMat3Usertype(tsd);
  registerMat4Usertype(tsd);

  // Top-level constructor functions and math utilities — these are
  // cheap to compile, so they stay in this TU.

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
