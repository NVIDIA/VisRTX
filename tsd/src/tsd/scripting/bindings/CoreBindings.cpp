// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ArrayHelpers.hpp"
#include "ObjectMethodBindings.hpp"
#include "ParameterHelpers.hpp"
#include "tsd/scene/Parameter.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/scene/Animation.hpp"
#include "tsd/scene/Object.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Array.hpp"
#include "tsd/scene/objects/Sampler.hpp"
#include "tsd/scripting/LuaBindings.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

static scene::ArrayRef createArrayFromLua(scene::Scene &scene,
    const std::string &typeStr,
    size_t items0,
    size_t items1,
    size_t items2)
{
  return scene.createArray(arrayTypeFromString(typeStr), items0, items1, items2);
}

static scene::ArrayRef createArrayFromLua(scene::Scene &scene,
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

template <typename F>
static auto makeForEach(F poolAccessor)
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

template <typename T>
static auto makeCreateBinding()
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

  auto objectType = tsd.new_usertype<scene::Object>(
      "Object", sol::no_constructor, "index", &scene::Object::index);

  registerObjectMethodsOn(
      objectType, [](scene::Object &o) -> scene::Object * { return &o; });

  tsd.new_usertype<scene::Scene>(
      "Scene",
      sol::constructors<scene::Scene()>(),
      // Object creation
      "createGeometry",
      makeCreateBinding<scene::Geometry>(),
      "createMaterial",
      makeCreateBinding<scene::Material>(),
      "createLight",
      makeCreateBinding<scene::Light>(),
      "createCamera",
      makeCreateBinding<scene::Camera>(),
      "createSampler",
      makeCreateBinding<scene::Sampler>(),
      "createVolume",
      makeCreateBinding<scene::Volume>(),
      "createSpatialField",
      makeCreateBinding<scene::SpatialField>(),
      "createSurface",
      [](scene::Scene &s,
          const std::string &name,
          scene::GeometryRef g,
          scene::MaterialRef m,
          sol::optional<sol::table> params) {
        auto ref = s.createSurface(name.c_str(), g, m);
        if (params)
          applyParameterTable(ref.data(), *params);
        return ref;
      },
      "createArray",
      sol::overload(
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
            return createArrayFromLua(
                s, typeStr, items0, items1, 0, data, st);
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
            return createArrayFromLua(
                s, typeStr, items0, items1, items2, data, st);
          }),
      // Object access
      "getGeometry",
      [](scene::Scene &s, size_t i) { return s.getObject<scene::Geometry>(i); },
      "getMaterial",
      [](scene::Scene &s, size_t i) { return s.getObject<scene::Material>(i); },
      "getLight",
      [](scene::Scene &s, size_t i) { return s.getObject<scene::Light>(i); },
      "getCamera",
      [](scene::Scene &s, size_t i) { return s.getObject<scene::Camera>(i); },
      "getSurface",
      [](scene::Scene &s, size_t i) { return s.getObject<scene::Surface>(i); },
      "getArray",
      [](scene::Scene &s, size_t i) { return s.getObject<scene::Array>(i); },
      "getVolume",
      [](scene::Scene &s, size_t i) { return s.getObject<scene::Volume>(i); },
      "getSampler",
      [](scene::Scene &s, size_t i) { return s.getObject<scene::Sampler>(i); },
      "getSpatialField",
      [](scene::Scene &s, size_t i) {
        return s.getObject<scene::SpatialField>(i);
      },
      // Object counts
      "numberOfObjects",
      [](scene::Scene &s, ANARIDataType type) -> size_t {
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
      [](scene::Scene &s, const std::string &name) {
        return s.addLayer(core::Token(name));
      },
      "layer",
      sol::overload(
          [](scene::Scene &s, const std::string &name) {
            return s.layer(core::Token(name));
          },
          [](scene::Scene &s, size_t i) { return s.layer(i); }),
      "numberOfLayers",
      &scene::Scene::numberOfLayers,
      "defaultLayer",
      &scene::Scene::defaultLayer,
      "defaultMaterial",
      &scene::Scene::defaultMaterial,
      // Node insertion
      "insertChildNode",
      [](scene::Scene &s, scene::LayerNodeRef parent, const std::string &name) {
        return s.insertChildNode(parent, name.c_str());
      },
      "insertChildTransformNode",
      [](scene::Scene &s,
          scene::LayerNodeRef parent,
          const math::mat4 &xfm,
          const std::string &name) {
        return s.insertChildTransformNode(parent, xfm, name.c_str());
      },
      "insertChildTransformArrayNode",
      sol::overload(
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
            return s.insertChildTransformArrayNode(
                parent, a.data(), name.c_str());
          }),
      // Object node insertion (adds objects to the renderable scene graph)
      "insertObjectNode",
      [](scene::Scene &s,
          scene::LayerNodeRef parent,
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
      [](scene::Scene &s, sol::object objArg) {
        auto *obj = extractObjectPtr(objArg);
        if (obj)
          s.removeObject(obj);
      },
      "removeAllObjects",
      &scene::Scene::removeAllObjects,
      // Layer removal
      "removeLayer",
      sol::overload(
          [](scene::Scene &s, const std::string &name) {
            s.removeLayer(core::Token(name));
          },
          [](scene::Scene &s, scene::Layer *layer) { s.removeLayer(layer); }),
      "removeAllLayers",
      &scene::Scene::removeAllLayers,
      // Layer active state
      "layerIsActive",
      [](scene::Scene &s, const std::string &name) {
        return s.layerIsActive(core::Token(name));
      },
      "setLayerActive",
      [](scene::Scene &s, const std::string &name, bool active) {
        s.setLayerActive(core::Token(name), active);
      },
      "setAllLayersActive",
      &scene::Scene::setAllLayersActive,
      "setOnlyLayerActive",
      [](scene::Scene &s, const std::string &name) {
        s.setOnlyLayerActive(core::Token(name));
      },
      "numberOfActiveLayers",
      &scene::Scene::numberOfActiveLayers,
      // Signal layer changes
      "signalLayerStructureChanged",
      [](scene::Scene &s, scene::Layer *l) {
        if (l)
          s.signalLayerStructureChanged(l);
      },
      "signalLayerTransformChanged",
      [](scene::Scene &s, scene::Layer *l) {
        if (l)
          s.signalLayerTransformChanged(l);
      },
      // Node removal
      "removeNode",
      sol::overload(
          [](scene::Scene &s, scene::LayerNodeRef obj) { s.removeNode(obj); },
          [](scene::Scene &s, scene::LayerNodeRef obj, bool deleteObjects) {
            s.removeNode(obj, deleteObjects);
          }),
      // Animation
      "addAnimation",
      sol::overload([](scene::Scene &s) { return s.addAnimation(); },
          [](scene::Scene &s, const std::string &name) {
            return s.addAnimation(name.c_str());
          }),
      "numberOfAnimations",
      &scene::Scene::numberOfAnimations,
      "animation",
      &scene::Scene::animation,
      "removeAnimation",
      &scene::Scene::removeAnimation,
      "removeAllAnimations",
      &scene::Scene::removeAllAnimations,
      "setAnimationTime",
      &scene::Scene::setAnimationTime,
      "getAnimationTime",
      &scene::Scene::getAnimationTime,
      "setAnimationIncrement",
      &scene::Scene::setAnimationIncrement,
      "getAnimationIncrement",
      &scene::Scene::getAnimationIncrement,
      "incrementAnimationTime",
      &scene::Scene::incrementAnimationTime,
      // Cleanup
      "removeUnusedObjects",
      &scene::Scene::removeUnusedObjects,
      "defragmentObjectStorage",
      &scene::Scene::defragmentObjectStorage,
      "cleanupScene",
      &scene::Scene::cleanupScene);

  tsd.new_usertype<scene::Animation>(
      "Animation",
      sol::no_constructor,
      "name",
      sol::property([](const scene::Animation &a) { return a.name(); },
          [](scene::Animation &a, const std::string &n) { a.name() = n; }),
      "info",
      [](const scene::Animation &a) { return a.info(); },
      "timeStepCount",
      &scene::Animation::timeStepCount,
      "update",
      &scene::Animation::update,
      "setAsTimeSteps",
      sol::overload(
          // Single parameter: anim:setAsTimeSteps(obj, "param", arrayRef)
          [](scene::Animation &a,
              sol::object obj,
              const std::string &param,
              scene::ArrayRef arr) {
            auto *o = extractObjectPtr(obj);
            if (!o)
              throw std::runtime_error(
                  "setAsTimeSteps: first argument must be a valid object");
            scene::TimeStepValues steps(arr);
            a.setAsTimeSteps(*o, core::Token(param), steps);
          },
          // Multi parameter: anim:setAsTimeSteps(obj, {"p1","p2"}, {arr1,arr2})
          [](scene::Animation &a,
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
            std::vector<scene::TimeStepValues> stepVec;
            for (size_t i = 1; i <= arrays.size(); i++)
              stepVec.emplace_back(arrays[i].get<scene::ArrayRef>());
            a.setAsTimeSteps(*o, paramVec, stepVec);
          }),
      "setAsTransformSteps",
      [](scene::Animation &a, scene::LayerNodeRef node, sol::table frames) {
        if (!node.valid())
          throw std::runtime_error(
              "setAsTransformSteps: node must be a valid LayerNode");
        std::vector<math::mat4> mats;
        mats.reserve(frames.size());
        for (size_t i = 1; i <= frames.size(); i++)
          mats.push_back(frames[i].get<math::mat4>());
        a.setAsTransformSteps(node, std::move(mats));
      });

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
