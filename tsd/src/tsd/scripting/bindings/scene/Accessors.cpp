// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

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

namespace tsd::scripting {

void registerSceneAccessors(sol::usertype<scene::Scene> &sceneType)
{
  sceneType["getGeometry"] = [](scene::Scene &s, size_t i) {
    return s.getObject<scene::Geometry>(i);
  };
  sceneType["getMaterial"] = [](scene::Scene &s, size_t i) {
    return s.getObject<scene::Material>(i);
  };
  sceneType["getLight"] = [](scene::Scene &s, size_t i) {
    return s.getObject<scene::Light>(i);
  };
  sceneType["getCamera"] = [](scene::Scene &s, size_t i) {
    return s.getObject<scene::Camera>(i);
  };
  sceneType["getSurface"] = [](scene::Scene &s, size_t i) {
    return s.getObject<scene::Surface>(i);
  };
  sceneType["getArray"] = [](scene::Scene &s, size_t i) {
    return s.getObject<scene::Array>(i);
  };
  sceneType["getVolume"] = [](scene::Scene &s, size_t i) {
    return s.getObject<scene::Volume>(i);
  };
  sceneType["getSampler"] = [](scene::Scene &s, size_t i) {
    return s.getObject<scene::Sampler>(i);
  };
  sceneType["getSpatialField"] = [](scene::Scene &s, size_t i) {
    return s.getObject<scene::SpatialField>(i);
  };
  sceneType["numberOfObjects"] =
      [](scene::Scene &s, anari::DataType type) -> size_t {
    return s.numberOfObjects(type);
  };
}

} // namespace tsd::scripting
