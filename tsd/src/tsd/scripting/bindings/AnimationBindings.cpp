// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Animation and AnimationManager sol2 usertypes. Lives in its own TU
// because each `new_usertype<>()` call is a heavy sol2 instantiation,
// and CoreBindings.cpp would otherwise be the build's critical path.

#include "AnimationBindings.hpp"
#include "ArrayHelpers.hpp"
#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/scene/Object.hpp"
#include "tsd/scripting/LuaBindings.hpp"

#include <sol/sol.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace tsd::scripting {

namespace {

std::vector<float> tableToFloats(sol::table t)
{
  std::vector<float> v(t.size());
  for (size_t i = 0; i < v.size(); i++)
    v[i] = t[i + 1].get<float>();
  return v;
}

template <typename Vec, size_t N>
std::vector<Vec> tableToVecs(sol::table t, const char *typeName)
{
  std::vector<Vec> v(t.size());
  for (size_t i = 0; i < v.size(); i++) {
    sol::object o = t[i + 1];
    if (o.is<Vec>()) {
      v[i] = o.as<Vec>();
    } else if (o.is<sol::table>()) {
      sol::table sub = o.as<sol::table>();
      if (sub.size() != N)
        throw std::runtime_error(
            std::string("expected ") + typeName + " or table of "
            + std::to_string(N) + " numbers");
      if constexpr (N == 2)
        v[i] = Vec(sub[1].get<float>(), sub[2].get<float>());
      else if constexpr (N == 3)
        v[i] = Vec(
            sub[1].get<float>(), sub[2].get<float>(), sub[3].get<float>());
      else if constexpr (N == 4)
        v[i] = Vec(sub[1].get<float>(),
            sub[2].get<float>(),
            sub[3].get<float>(),
            sub[4].get<float>());
    } else {
      throw std::runtime_error(
          std::string("expected ") + typeName + " or table of "
          + std::to_string(N) + " numbers");
    }
  }
  return v;
}

} // namespace

void registerAnimationBindings(sol::table &tsd)
{
  tsd.new_usertype<tsd::animation::Animation>(
      "Animation",
      sol::no_constructor,
      "name",
      &tsd::animation::Animation::name,
      "addObjectParameterBinding",
      [](tsd::animation::Animation &a,
          sol::object target,
          const std::string &param,
          const std::string &typeStr,
          sol::table dataTable,
          sol::table timeBaseTable,
          sol::optional<std::string> interpStr) {
        auto *obj = extractObjectPtr(target);
        if (!obj)
          throw std::runtime_error(
              "addObjectParameterBinding: invalid target object");

        auto interp = tsd::animation::InterpolationRule::LINEAR;
        if (interpStr && *interpStr == "step")
          interp = tsd::animation::InterpolationRule::STEP;
        else if (interpStr && *interpStr == "slerp")
          interp = tsd::animation::InterpolationRule::SLERP;

        auto dataType = arrayTypeFromString(typeStr);
        auto tb = tableToFloats(timeBaseTable);
        size_t count = std::min<size_t>(dataTable.size(), tb.size());

        if (anari::isObject(dataType)) {
          std::vector<scene::Object *> ptrs(count);
          for (size_t i = 0; i < count; i++) {
            ptrs[i] = extractObjectPtr(dataTable[i + 1]);
            if (!ptrs[i])
              throw std::runtime_error(
                  "addObjectParameterBinding: invalid object at index "
                  + std::to_string(i));
          }
          a.addObjectParameterBinding(
              obj, core::Token(param), dataType, ptrs.data(), tb.data(),
              count, interp);
        } else {
          // Value types — decode into a typed vector, then pass as void*
          auto addValues = [&](auto *typed, auto vec) {
            (void)typed;
            a.addObjectParameterBinding(
                obj, core::Token(param), dataType, vec.data(), tb.data(),
                count, interp);
          };

          switch (dataType) {
          case ANARI_FLOAT32:
            addValues((float *)nullptr, tableToFloats(dataTable));
            break;
          case ANARI_FLOAT32_VEC2:
            addValues((math::float2 *)nullptr,
                tableToVecs<math::float2, 2>(dataTable, "float2"));
            break;
          case ANARI_FLOAT32_VEC3:
            addValues((math::float3 *)nullptr,
                tableToVecs<math::float3, 3>(dataTable, "float3"));
            break;
          case ANARI_FLOAT32_VEC4:
            addValues((math::float4 *)nullptr,
                tableToVecs<math::float4, 4>(dataTable, "float4"));
            break;
          case ANARI_FLOAT32_MAT4: {
            std::vector<math::mat4> v(count);
            for (size_t i = 0; i < count; i++) {
              sol::object o = dataTable[i + 1];
              if (o.is<math::mat4>())
                v[i] = o.as<math::mat4>();
              else
                throw std::runtime_error(
                    "addObjectParameterBinding: expected mat4 at index "
                    + std::to_string(i));
            }
            a.addObjectParameterBinding(
                obj, core::Token(param), dataType, v.data(), tb.data(),
                count, interp);
            break;
          }
          case ANARI_INT32: {
            std::vector<int32_t> v(count);
            for (size_t i = 0; i < count; i++)
              v[i] = dataTable[i + 1].get<int32_t>();
            a.addObjectParameterBinding(
                obj, core::Token(param), dataType, v.data(), tb.data(),
                count, interp);
            break;
          }
          case ANARI_UINT32: {
            std::vector<uint32_t> v(count);
            for (size_t i = 0; i < count; i++)
              v[i] = dataTable[i + 1].get<uint32_t>();
            a.addObjectParameterBinding(
                obj, core::Token(param), dataType, v.data(), tb.data(),
                count, interp);
            break;
          }
          default:
            throw std::runtime_error(
                "addObjectParameterBinding: unsupported data type '"
                + typeStr + "'");
          }
        }
      },
      "addTransformBinding",
      [](tsd::animation::Animation &a,
          scene::LayerNodeRef node,
          sol::table timeBaseTable,
          sol::table rotTable,
          sol::table transTable,
          sol::table scaleTable) {
        if (!node.valid())
          throw std::runtime_error("addTransformBinding: node must be valid");
        auto tb = tableToFloats(timeBaseTable);
        auto rot = tableToVecs<math::float4, 4>(rotTable, "float4");
        auto trans = tableToVecs<math::float3, 3>(transTable, "float3");
        auto scale = tableToVecs<math::float3, 3>(scaleTable, "float3");
        a.addTransformBinding(
            node, tb.data(), rot.data(), trans.data(), scale.data(), tb.size());
      });
}

void registerAnimationManagerBindings(sol::state &lua)
{
  using SA = tsd::animation::AnimationManager;
  sol::table tsd = lua["tsd"];

  tsd.new_usertype<SA>(
      "AnimationManager",
      sol::no_constructor,
      "addAnimation",
      sol::overload(
          [](SA &sa) -> tsd::animation::Animation & {
            return sa.addAnimation();
          },
          [](SA &sa, const std::string &name) -> tsd::animation::Animation & {
            return sa.addAnimation(name);
          }),
      "animations",
      [](SA &sa) -> std::vector<tsd::animation::Animation> & {
        return sa.animations();
      },
      "numberOfAnimations",
      [](SA &sa) -> size_t { return sa.animations().size(); },
      "removeAnimation",
      &SA::removeAnimation,
      "removeAllAnimations",
      &SA::removeAllAnimations,
      "setAnimationTime",
      &SA::setAnimationTime,
      "getAnimationTime",
      &SA::getAnimationTime,
      "setAnimationIncrement",
      &SA::setAnimationIncrement,
      "getAnimationIncrement",
      &SA::getAnimationIncrement,
      "incrementAnimationTime",
      &SA::incrementAnimationTime,
      "getAnimationTotalFrames",
      &SA::getAnimationTotalFrames,
      "setAnimationTotalFrames",
      &SA::setAnimationTotalFrames,
      "getAnimationFPS",
      &SA::getAnimationFPS,
      "setAnimationFPS",
      &SA::setAnimationFPS,
      "getAnimationFrame",
      &SA::getAnimationFrame,
      "setAnimationFrame",
      &SA::setAnimationFrame,
      "incrementAnimationFrame",
      &SA::incrementAnimationFrame);
}

} // namespace tsd::scripting
