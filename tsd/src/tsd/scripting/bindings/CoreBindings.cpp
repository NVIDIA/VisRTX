// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ArrayHelpers.hpp"
#include "ObjectMethodBindings.hpp"
#include "ParameterHelpers.hpp"
#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/scene/Object.hpp"
#include "tsd/scene/Parameter.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Array.hpp"
#include "tsd/scene/objects/Sampler.hpp"
#include "tsd/scripting/LuaBindings.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

// Animation helpers ///////////////////////////////////////////////////////////

static std::vector<float> tableToFloats(sol::table t)
{
  std::vector<float> v(t.size());
  for (size_t i = 0; i < v.size(); i++)
    v[i] = t[i + 1].get<float>();
  return v;
}

template <typename Vec, size_t N>
static std::vector<Vec> tableToVecs(sol::table t, const char *typeName)
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

static scene::ArrayRef createArrayFromLua(scene::Scene &scene,
    const std::string &typeStr,
    size_t items0,
    size_t items1,
    size_t items2)
{
  return scene.createArray(
      arrayTypeFromString(typeStr), items0, items1, items2);
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
      [](scene::Scene &s, anari::DataType type) -> size_t {
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
      // Cleanup
      "removeUnusedObjects",
      &scene::Scene::removeUnusedObjects,
      "defragmentObjectStorage",
      &scene::Scene::defragmentObjectStorage,
      "cleanupScene",
      &scene::Scene::cleanupScene);

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
