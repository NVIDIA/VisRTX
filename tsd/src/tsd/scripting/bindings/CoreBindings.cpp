// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AnimationBindings.hpp"
#include "ObjectUsertype.hpp"
#include "SceneUsertype.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/scene/Parameter.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scripting/LuaBindings.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

#include <memory>

namespace tsd::scripting {

void registerContextBindings(sol::state &lua)
{
  sol::table tsd = lua["tsd"];

  tsd.new_usertype<core::Token>("Token",
      sol::constructors<core::Token(), core::Token(const char *)>(),
      "str",
      &core::Token::str,
      "empty",
      &core::Token::empty,
      sol::meta_function::to_string,
      &core::Token::str,
      sol::meta_function::equal_to,
      [](const core::Token &a, const core::Token &b) { return a == b; });

  // Read-only from Lua; values are set through Object
  tsd.new_usertype<scene::Parameter>(
      "Parameter",
      sol::no_constructor,
      "name",
      [](const scene::Parameter &p) { return p.name().str(); },
      "description",
      &scene::Parameter::description,
      "isEnabled",
      &scene::Parameter::isEnabled);

  // scene::Object usertype lives in ObjectUsertype.cpp.
  registerObjectUsertype(tsd);

  // scene::Scene usertype is created here with just its constructor;
  // each method group is layered on by a per-group TU under
  // bindings/scene/ so they parallelise across cores.
  auto sceneType = tsd.new_usertype<scene::Scene>(
      "Scene", sol::constructors<scene::Scene()>());

  registerSceneCreators(sceneType);
  registerSceneAccessors(sceneType);
  registerSceneIteration(sceneType);
  registerSceneLayers(sceneType);
  registerSceneNodes(sceneType);

  // Animation usertype lives in AnimationBindings.cpp.
  registerAnimationBindings(tsd);

  tsd["createScene"] = []() { return std::make_unique<scene::Scene>(); };

  // ANARI data type constants
  tsd["GEOMETRY"] = ANARI_GEOMETRY;
  tsd["MATERIAL"] = ANARI_MATERIAL;
  tsd["LIGHT"] = ANARI_LIGHT;
  tsd["CAMERA"] = ANARI_CAMERA;
  tsd["SURFACE"] = ANARI_SURFACE;
  tsd["VOLUME"] = ANARI_VOLUME;
  tsd["SAMPLER"] = ANARI_SAMPLER;
  tsd["ARRAY"] = ANARI_ARRAY;
  tsd["SPATIAL_FIELD"] = ANARI_SPATIAL_FIELD;
}

} // namespace tsd::scripting
