// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ArrayHelpers.hpp"
#include "ObjectMethodBindings.hpp"
#include "ParameterHelpers.hpp"
#include "tsd/scripting/LuaBindings.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/core/Parameter.hpp"
#include "tsd/core/scene/Animation.hpp"
#include "tsd/core/scene/Object.hpp"
#include "tsd/core/scene/Scene.hpp"
#include "tsd/core/scene/objects/Array.hpp"
#include "tsd/core/scene/objects/Sampler.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

static core::Object *extractObjectPtr(sol::object luaObj)
{
  if (luaObj.is<core::GeometryRef>()) {
    auto ref = luaObj.as<core::GeometryRef>();
    return ref.valid() ? ref.data() : nullptr;
  }
  if (luaObj.is<core::MaterialRef>()) {
    auto ref = luaObj.as<core::MaterialRef>();
    return ref.valid() ? ref.data() : nullptr;
  }
  if (luaObj.is<core::LightRef>()) {
    auto ref = luaObj.as<core::LightRef>();
    return ref.valid() ? ref.data() : nullptr;
  }
  if (luaObj.is<core::CameraRef>()) {
    auto ref = luaObj.as<core::CameraRef>();
    return ref.valid() ? ref.data() : nullptr;
  }
  if (luaObj.is<core::SamplerRef>()) {
    auto ref = luaObj.as<core::SamplerRef>();
    return ref.valid() ? ref.data() : nullptr;
  }
  if (luaObj.is<core::SurfaceRef>()) {
    auto ref = luaObj.as<core::SurfaceRef>();
    return ref.valid() ? ref.data() : nullptr;
  }
  if (luaObj.is<core::VolumeRef>()) {
    auto ref = luaObj.as<core::VolumeRef>();
    return ref.valid() ? ref.data() : nullptr;
  }
  if (luaObj.is<core::SpatialFieldRef>()) {
    auto ref = luaObj.as<core::SpatialFieldRef>();
    return ref.valid() ? ref.data() : nullptr;
  }
  if (luaObj.is<core::ArrayRef>()) {
    auto ref = luaObj.as<core::ArrayRef>();
    return ref.valid() ? ref.data() : nullptr;
  }
  if (luaObj.is<core::Object *>()) {
    return luaObj.as<core::Object *>();
  }
  return nullptr;
}

template <typename F>
static auto makeForEach(F poolAccessor)
{
  return [poolAccessor](core::Scene &s, sol::function fn) {
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

template <typename T>
static auto makeCreateBinding()
{
  return [](core::Scene &s,
             const std::string &subtype,
             sol::optional<sol::table> params) {
    auto ref = s.createObject<T>(core::Token(subtype));
    if (params)
      applyParameterTable(ref.data(), *params);
    return ref;
  };
}


void registerCoreBindings(sol::state &lua)
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
  tsd.new_usertype<core::Parameter>("Parameter",
      sol::no_constructor,
      "name",
      [](const core::Parameter &p) { return p.name().str(); },
      "description",
      &core::Parameter::description,
      "isEnabled",
      &core::Parameter::isEnabled);

  auto objectType = tsd.new_usertype<core::Object>("Object",
      sol::no_constructor,
      "index",
      &core::Object::index);

  registerObjectMethodsOn(
      objectType, [](core::Object &o) -> core::Object * { return &o; });

  tsd.new_usertype<core::Scene>("Scene",
      sol::constructors<core::Scene()>(),
      // Object creation
      "createGeometry", makeCreateBinding<core::Geometry>(),
      "createMaterial", makeCreateBinding<core::Material>(),
      "createLight", makeCreateBinding<core::Light>(),
      "createCamera", makeCreateBinding<core::Camera>(),
      "createSampler", makeCreateBinding<core::Sampler>(),
      "createVolume", makeCreateBinding<core::Volume>(),
      "createSpatialField", makeCreateBinding<core::SpatialField>(),
      "createSurface",
      [](core::Scene &s,
          const std::string &name,
          core::GeometryRef g,
          core::MaterialRef m,
          sol::optional<sol::table> params) {
        auto ref = s.createSurface(name.c_str(), g, m);
        if (params)
          applyParameterTable(ref.data(), *params);
        return ref;
      },
      "createArray",
      sol::overload(
          [](core::Scene &s, const std::string &typeStr, size_t items0) {
            return s.createArray(arrayTypeFromString(typeStr), items0);
          },
          [](core::Scene &s,
              const std::string &typeStr,
              size_t items0,
              size_t items1) {
            return s.createArray(arrayTypeFromString(typeStr), items0, items1);
          },
          [](core::Scene &s,
              const std::string &typeStr,
              size_t items0,
              size_t items1,
              size_t items2) {
            return s.createArray(
                arrayTypeFromString(typeStr), items0, items1, items2);
          }),
      // Object access
      "getGeometry",
      [](core::Scene &s, size_t i) { return s.getObject<core::Geometry>(i); },
      "getMaterial",
      [](core::Scene &s, size_t i) { return s.getObject<core::Material>(i); },
      "getLight",
      [](core::Scene &s, size_t i) { return s.getObject<core::Light>(i); },
      "getCamera",
      [](core::Scene &s, size_t i) { return s.getObject<core::Camera>(i); },
      "getSurface",
      [](core::Scene &s, size_t i) {
        return s.getObject<core::Surface>(i);
      },
      "getArray",
      [](core::Scene &s, size_t i) { return s.getObject<core::Array>(i); },
      "getVolume",
      [](core::Scene &s, size_t i) { return s.getObject<core::Volume>(i); },
      "getSampler",
      [](core::Scene &s, size_t i) { return s.getObject<core::Sampler>(i); },
      "getSpatialField",
      [](core::Scene &s, size_t i) {
        return s.getObject<core::SpatialField>(i);
      },
      // Object counts
      "numberOfObjects",
      [](core::Scene &s, ANARIDataType type) -> size_t {
        return s.numberOfObjects(type);
      },
      // Iteration over objects
      "forEachGeometry",
      makeForEach([](auto &db) -> auto & { return db.geometry; }),
      "forEachMaterial",
      makeForEach([](auto &db) -> auto & { return db.material; }),
      "forEachSurface",
      makeForEach([](auto &db) -> auto & { return db.surface; }),
      "forEachLight",
      makeForEach([](auto &db) -> auto & { return db.light; }),
      "forEachCamera",
      makeForEach([](auto &db) -> auto & { return db.camera; }),
      "forEachVolume",
      makeForEach([](auto &db) -> auto & { return db.volume; }),
      "forEachSpatialField",
      makeForEach([](auto &db) -> auto & { return db.field; }),
      "forEachSampler",
      makeForEach([](auto &db) -> auto & { return db.sampler; }),
      "forEachArray",
      makeForEach([](auto &db) -> auto & { return db.array; }),
      // Layers
      "addLayer",
      [](core::Scene &s, const std::string &name) {
        return s.addLayer(core::Token(name));
      },
      "layer",
      sol::overload(
          [](core::Scene &s, const std::string &name) {
            return s.layer(core::Token(name));
          },
          [](core::Scene &s, size_t i) { return s.layer(i); }),
      "numberOfLayers",
      &core::Scene::numberOfLayers,
      "defaultLayer",
      &core::Scene::defaultLayer,
      "defaultMaterial",
      &core::Scene::defaultMaterial,
      // Node insertion
      "insertChildNode",
      [](core::Scene &s, core::LayerNodeRef parent, const std::string &name) {
        return s.insertChildNode(parent, name.c_str());
      },
      "insertChildTransformNode",
      [](core::Scene &s,
          core::LayerNodeRef parent,
          const math::mat4 &xfm,
          const std::string &name) {
        return s.insertChildTransformNode(parent, xfm, name.c_str());
      },
      "insertChildTransformArrayNode",
      sol::overload(
          [](core::Scene &s,
              core::LayerNodeRef parent,
              core::Array &a,
              const std::string &name) {
            return s.insertChildTransformArrayNode(parent, &a, name.c_str());
          },
          [](core::Scene &s,
              core::LayerNodeRef parent,
              core::ArrayRef a,
              const std::string &name) {
            if (!a)
              throw std::runtime_error(
                  "insertChildTransformArrayNode: invalid array");
            return s.insertChildTransformArrayNode(
                parent, a.data(), name.c_str());
          }),
      // Object node insertion (adds objects to the renderable scene graph)
      "insertObjectNode",
      [](core::Scene &s,
          core::LayerNodeRef parent,
          sol::object objArg,
          sol::optional<std::string> name) {
        auto *obj = extractObjectPtr(objArg);
        if (!obj)
          throw std::runtime_error("insertObjectNode: invalid object argument");
        return s.insertChildObjectNode(
            parent, obj->type(), obj->index(), name.value_or("").c_str());
      },
      // Object removal
      "removeObject",
      [](core::Scene &s, sol::object objArg) {
        auto *obj = extractObjectPtr(objArg);
        if (obj)
          s.removeObject(obj);
      },
      "removeAllObjects",
      &core::Scene::removeAllObjects,
      // Layer removal
      "removeLayer",
      sol::overload(
          [](core::Scene &s, const std::string &name) {
            s.removeLayer(core::Token(name));
          },
          [](core::Scene &s, core::Layer *layer) { s.removeLayer(layer); }),
      "removeAllLayers",
      &core::Scene::removeAllLayers,
      // Layer active state
      "layerIsActive",
      [](core::Scene &s, const std::string &name) {
        return s.layerIsActive(core::Token(name));
      },
      "setLayerActive",
      [](core::Scene &s, const std::string &name, bool active) {
        s.setLayerActive(core::Token(name), active);
      },
      "setAllLayersActive",
      &core::Scene::setAllLayersActive,
      "setOnlyLayerActive",
      [](core::Scene &s, const std::string &name) {
        s.setOnlyLayerActive(core::Token(name));
      },
      "numberOfActiveLayers",
      &core::Scene::numberOfActiveLayers,
      // Signal layer change (needed after modifying transforms)
      "signalLayerChange",
      [](core::Scene &s, core::Layer *l) {
        if (l)
          s.signalLayerChange(l);
      },
      // Node removal
      "removeNode",
      sol::overload(
          [](core::Scene &s, core::LayerNodeRef obj) {
            s.removeNode(obj);
          },
          [](core::Scene &s, core::LayerNodeRef obj, bool deleteObjects) {
            s.removeNode(obj, deleteObjects);
          }),
      // Animation
      "addAnimation",
      sol::overload(
          [](core::Scene &s) { return s.addAnimation(); },
          [](core::Scene &s, const std::string &name) {
            return s.addAnimation(name.c_str());
          }),
      "numberOfAnimations",
      &core::Scene::numberOfAnimations,
      "animation",
      &core::Scene::animation,
      "removeAnimation",
      &core::Scene::removeAnimation,
      "removeAllAnimations",
      &core::Scene::removeAllAnimations,
      "setAnimationTime",
      &core::Scene::setAnimationTime,
      "getAnimationTime",
      &core::Scene::getAnimationTime,
      "setAnimationIncrement",
      &core::Scene::setAnimationIncrement,
      "getAnimationIncrement",
      &core::Scene::getAnimationIncrement,
      "incrementAnimationTime",
      &core::Scene::incrementAnimationTime,
      // Cleanup
      "removeUnusedObjects",
      &core::Scene::removeUnusedObjects,
      "defragmentObjectStorage",
      &core::Scene::defragmentObjectStorage,
      "cleanupScene",
      &core::Scene::cleanupScene);

  tsd.new_usertype<core::Animation>("Animation",
      sol::no_constructor,
      "name",
      sol::property(
          [](const core::Animation &a) { return a.name(); },
          [](core::Animation &a, const std::string &n) { a.name() = n; }),
      "info",
      [](const core::Animation &a) { return a.info(); },
      "timeStepCount",
      &core::Animation::timeStepCount,
      "update",
      &core::Animation::update,
      "setAsTimeSteps",
      sol::overload(
          // Single parameter: anim:setAsTimeSteps(obj, "param", arrayRef)
          [](core::Animation &a,
              sol::object obj,
              const std::string &param,
              core::ArrayRef arr) {
            auto *o = extractObjectPtr(obj);
            if (!o)
              throw std::runtime_error(
                  "setAsTimeSteps: first argument must be a valid object");
            core::TimeStepValues steps(arr);
            a.setAsTimeSteps(*o, core::Token(param), steps);
          },
          // Multi parameter: anim:setAsTimeSteps(obj, {"p1","p2"}, {arr1,arr2})
          [](core::Animation &a,
              sol::object obj,
              sol::table params,
              sol::table arrays) {
            auto *o = extractObjectPtr(obj);
            if (!o)
              throw std::runtime_error(
                  "setAsTimeSteps: first argument must be a valid object");
            std::vector<core::Token> paramVec;
            for (size_t i = 1; i <= params.size(); i++)
              paramVec.emplace_back(params[i].get<std::string>().c_str());
            std::vector<core::TimeStepValues> stepVec;
            for (size_t i = 1; i <= arrays.size(); i++)
              stepVec.emplace_back(arrays[i].get<core::ArrayRef>());
            a.setAsTimeSteps(*o, paramVec, stepVec);
          }));

  tsd["createScene"] = []() { return std::make_unique<core::Scene>(); };

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
