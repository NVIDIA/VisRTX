// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ArrayHelpers.hpp"
#include "../SceneUsertype.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Array.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

#include <stdexcept>

namespace tsd::scripting {

void registerSceneNodes(sol::usertype<scene::Scene> &sceneType)
{
  sceneType["insertChildNode"] = [](scene::Scene &s,
                                     scene::LayerNodeRef parent,
                                     const std::string &name) {
    return s.insertChildNode(parent, name.c_str());
  };

  sceneType["insertChildTransformNode"] = [](scene::Scene &s,
                                              scene::LayerNodeRef parent,
                                              const math::mat4 &xfm,
                                              const std::string &name) {
    return s.insertChildTransformNode(parent, xfm, name.c_str());
  };

  sceneType["insertChildTransformArrayNode"] = sol::overload(
      [](scene::Scene &s,
          scene::LayerNodeRef parent,
          scene::Array &a,
          const std::string &name) {
        return s.insertChildTransformArrayNode(parent, &a, name.c_str());
      },
      [](scene::Scene &s,
          scene::LayerNodeRef parent,
          scene::ArrayRef a,
          const std::string &name) {
        if (!a)
          throw std::runtime_error(
              "insertChildTransformArrayNode: invalid array");
        return s.insertChildTransformArrayNode(parent, a.data(), name.c_str());
      });

  sceneType["insertObjectNode"] = [](scene::Scene &s,
                                      scene::LayerNodeRef parent,
                                      sol::object objArg,
                                      sol::optional<std::string> name) {
    auto *obj = extractObjectPtr(objArg);
    if (!obj)
      throw std::runtime_error("insertObjectNode: invalid object argument");
    return s.insertChildObjectNode(
        parent, obj->type(), obj->index(), name.value_or("").c_str());
  };

  sceneType["removeObject"] = [](scene::Scene &s, sol::object objArg) {
    auto *obj = extractObjectPtr(objArg);
    if (obj)
      s.removeObject(obj);
  };

  sceneType["removeAllObjects"] = &scene::Scene::removeAllObjects;

  sceneType["removeNode"] = sol::overload(
      [](scene::Scene &s, scene::LayerNodeRef obj) { s.removeNode(obj); },
      [](scene::Scene &s, scene::LayerNodeRef obj, bool deleteObjects) {
        s.removeNode(obj, deleteObjects);
      });

  sceneType["removeUnusedObjects"] = &scene::Scene::removeUnusedObjects;
  sceneType["defragmentObjectStorage"] = &scene::Scene::defragmentObjectStorage;
  sceneType["cleanupScene"] = &scene::Scene::cleanupScene;
}

} // namespace tsd::scripting
