// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../SceneUsertype.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

namespace {

template <typename F>
auto makeForEach(F poolAccessor)
{
  return [poolAccessor](scene::Scene &s, sol::function fn) {
    const auto &pool = poolAccessor(s.objectDB());
    for (size_t i = 0; i < pool.capacity(); i++) {
      if (!pool.slot_empty(i)) {
        sol::object result = fn(pool.at(i));
        if (result.is<bool>() && !result.as<bool>())
          break;
      }
    }
  };
}

} // namespace

void registerSceneIteration(sol::usertype<scene::Scene> &sceneType)
{
  sceneType["forEachGeometry"] =
      makeForEach([](auto &db) -> auto & { return db.geometry; });
  sceneType["forEachMaterial"] =
      makeForEach([](auto &db) -> auto & { return db.material; });
  sceneType["forEachSurface"] =
      makeForEach([](auto &db) -> auto & { return db.surface; });
  sceneType["forEachLight"] =
      makeForEach([](auto &db) -> auto & { return db.light; });
  sceneType["forEachCamera"] =
      makeForEach([](auto &db) -> auto & { return db.camera; });
  sceneType["forEachVolume"] =
      makeForEach([](auto &db) -> auto & { return db.volume; });
  sceneType["forEachSpatialField"] =
      makeForEach([](auto &db) -> auto & { return db.field; });
  sceneType["forEachSampler"] =
      makeForEach([](auto &db) -> auto & { return db.sampler; });
  sceneType["forEachArray"] =
      makeForEach([](auto &db) -> auto & { return db.array; });
}

} // namespace tsd::scripting
