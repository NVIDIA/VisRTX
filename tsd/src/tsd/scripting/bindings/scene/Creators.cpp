// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ArrayHelpers.hpp"
#include "../ParameterHelpers.hpp"
#include "../SceneUsertype.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Array.hpp"
#include "tsd/scene/objects/Camera.hpp"
#include "tsd/scene/objects/Geometry.hpp"
#include "tsd/scene/objects/Light.hpp"
#include "tsd/scene/objects/Material.hpp"
#include "tsd/scene/objects/Sampler.hpp"
#include "tsd/scene/objects/SpatialField.hpp"
#include "tsd/scene/objects/Surface.hpp"
#include "tsd/scene/objects/Volume.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

#include <stdexcept>

namespace tsd::scripting {

namespace {

scene::ArrayRef createArrayFromLua(scene::Scene &scene,
    const std::string &typeStr,
    size_t items0,
    size_t items1,
    size_t items2)
{
  return scene.createArray(
      arrayTypeFromString(typeStr), items0, items1, items2);
}

scene::ArrayRef createArrayFromLua(scene::Scene &scene,
    const std::string &typeStr,
    size_t items0,
    size_t items1,
    size_t items2,
    sol::table data,
    sol::this_state s)
{
  const auto elemType = arrayTypeFromString(typeStr);
  const bool isObj = anari::isObject(elemType);

  if (items0 == 0) {
    if (isObj) {
      items0 = data.size();
    } else {
      inferArrayDimsFromLuaData(data, elemType, items0, items1, items2);
    }
  }

  auto arr = scene.createArray(elemType, items0, items1, items2);
  if (!arr.valid())
    throw std::runtime_error("createArray: failed to create array");

  if (isObj)
    arraySetObjectsFromLua(*arr.data(), data);
  else
    arraySetDataFromLua(*arr.data(), data, s);

  return arr;
}

template <typename T>
auto makeCreateBinding()
{
  return [](scene::Scene &s,
             const std::string &subtype,
             sol::optional<sol::table> params) {
    auto ref = s.createObject<T>(core::Token(subtype));
    if (params)
      applyParameterTable(ref.data(), *params);
    return ref;
  };
}

} // namespace

void registerSceneCreators(sol::usertype<scene::Scene> &sceneType)
{
  sceneType["createGeometry"] = makeCreateBinding<scene::Geometry>();
  sceneType["createMaterial"] = makeCreateBinding<scene::Material>();
  sceneType["createLight"] = makeCreateBinding<scene::Light>();
  sceneType["createCamera"] = makeCreateBinding<scene::Camera>();
  sceneType["createSampler"] = makeCreateBinding<scene::Sampler>();
  sceneType["createVolume"] = makeCreateBinding<scene::Volume>();
  sceneType["createSpatialField"] = makeCreateBinding<scene::SpatialField>();

  sceneType["createSurface"] = [](scene::Scene &s,
                                   const std::string &name,
                                   scene::GeometryRef g,
                                   scene::MaterialRef m,
                                   sol::optional<sol::table> params) {
    auto ref = s.createSurface(name.c_str(), g, m);
    if (params)
      applyParameterTable(ref.data(), *params);
    return ref;
  };

  sceneType["createArray"] = sol::overload(
      // (typeStr, table) — infer dims from data
      [](scene::Scene &s,
          const std::string &typeStr,
          sol::table data,
          sol::this_state st) {
        return createArrayFromLua(s, typeStr, 0, 0, 0, data, st);
      },
      // (typeStr, items0) — empty 1D
      [](scene::Scene &s, const std::string &typeStr, size_t items0) {
        return createArrayFromLua(s, typeStr, items0, 0, 0);
      },
      // (typeStr, items0, table) — 1D with data
      [](scene::Scene &s,
          const std::string &typeStr,
          size_t items0,
          sol::table data,
          sol::this_state st) {
        return createArrayFromLua(s, typeStr, items0, 0, 0, data, st);
      },
      // (typeStr, items0, items1) — empty 2D
      [](scene::Scene &s,
          const std::string &typeStr,
          size_t items0,
          size_t items1) {
        return createArrayFromLua(s, typeStr, items0, items1, 0);
      },
      // (typeStr, items0, items1, table) — 2D with data
      [](scene::Scene &s,
          const std::string &typeStr,
          size_t items0,
          size_t items1,
          sol::table data,
          sol::this_state st) {
        return createArrayFromLua(s, typeStr, items0, items1, 0, data, st);
      },
      // (typeStr, items0, items1, items2) — empty 3D
      [](scene::Scene &s,
          const std::string &typeStr,
          size_t items0,
          size_t items1,
          size_t items2) {
        return createArrayFromLua(s, typeStr, items0, items1, items2);
      },
      // (typeStr, items0, items1, items2, table) — 3D with data
      [](scene::Scene &s,
          const std::string &typeStr,
          size_t items0,
          size_t items1,
          size_t items2,
          sol::table data,
          sol::this_state st) {
        return createArrayFromLua(s, typeStr, items0, items1, items2, data, st);
      });
}

} // namespace tsd::scripting
