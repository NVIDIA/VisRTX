// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../SceneUsertype.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/scene/Layer.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerSceneLayers(sol::usertype<scene::Scene> &sceneType)
{
  sceneType["addLayer"] = [](scene::Scene &s, const std::string &name) {
    return s.addLayer(core::Token(name));
  };

  sceneType["layer"] = sol::overload(
      [](scene::Scene &s, const std::string &name) {
        return s.layer(core::Token(name));
      },
      [](scene::Scene &s, size_t i) { return s.layer(i); });

  sceneType["numberOfLayers"] = &scene::Scene::numberOfLayers;
  sceneType["defaultLayer"] = &scene::Scene::defaultLayer;
  sceneType["defaultMaterial"] = &scene::Scene::defaultMaterial;

  sceneType["removeLayer"] = sol::overload(
      [](scene::Scene &s, const std::string &name) {
        s.removeLayer(core::Token(name));
      },
      [](scene::Scene &s, scene::Layer *layer) { s.removeLayer(layer); });

  sceneType["removeAllLayers"] = &scene::Scene::removeAllLayers;

  sceneType["layerIsActive"] = [](scene::Scene &s, const std::string &name) {
    return s.layerIsActive(core::Token(name));
  };

  sceneType["setLayerActive"] =
      [](scene::Scene &s, const std::string &name, bool active) {
        s.setLayerActive(core::Token(name), active);
      };

  sceneType["setAllLayersActive"] = &scene::Scene::setAllLayersActive;

  sceneType["setOnlyLayerActive"] =
      [](scene::Scene &s, const std::string &name) {
        s.setOnlyLayerActive(core::Token(name));
      };

  sceneType["numberOfActiveLayers"] = &scene::Scene::numberOfActiveLayers;

  sceneType["signalLayerStructureChanged"] =
      [](scene::Scene &s, scene::Layer *l) {
        if (l)
          s.signalLayerStructureChanged(l);
      };

  sceneType["signalLayerTransformChanged"] =
      [](scene::Scene &s, scene::Layer *l) {
        if (l)
          s.signalLayerTransformChanged(l);
      };
}

} // namespace tsd::scripting
