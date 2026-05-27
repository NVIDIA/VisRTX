// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ArrayHelpers.hpp"
#include "ObjectRefBindings.hpp"
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
#include "tsd/scripting/LuaBindings.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

#include <functional>

namespace tsd::scripting {

void registerObjectBindings(sol::state &lua)
{
  sol::table tsd = lua["tsd"];

  // Concrete Object types — metatables are keyed by C++ type_index; the
  // per-type Ref registrations below overwrite the named `tsd["Name"]`
  // entries so Lua code lands on the Ref usertype, not the Object one.

  tsd.new_usertype<scene::Geometry>("Geometry",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<scene::Object>());

  tsd.new_usertype<scene::Material>("Material",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<scene::Object>());

  tsd.new_usertype<scene::Light>("Light",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<scene::Object>());

  tsd.new_usertype<scene::Camera>("Camera",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<scene::Object>());

  tsd.new_usertype<scene::Surface>(
      "Surface",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<scene::Object>(),
      sol::meta_function::equal_to,
      [](const scene::Surface &a, const scene::Surface &b) { return &a == &b; },
      sol::meta_function::less_than,
      [](const scene::Surface &a, const scene::Surface &b) {
        return std::less<const scene::Surface *>{}(&a, &b);
      },
      "geometry",
      [](scene::Surface &s) {
        return s.parameterValueAsObject<scene::Geometry>("geometry");
      },
      "material",
      [](scene::Surface &s) {
        return s.parameterValueAsObject<scene::Material>("material");
      });

  tsd.new_usertype<scene::Volume>("Volume",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<scene::Object>(),
      "spatialField",
      [](scene::Volume &v) {
        return v.parameterValueAsObject<scene::SpatialField>("value");
      });

  tsd.new_usertype<scene::Sampler>("Sampler",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<scene::Object>());

  tsd.new_usertype<scene::SpatialField>(
      "SpatialField",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<scene::Object>(),
      sol::meta_function::equal_to,
      [](const scene::SpatialField &a, const scene::SpatialField &b) {
        return &a == &b;
      },
      sol::meta_function::less_than,
      [](const scene::SpatialField &a, const scene::SpatialField &b) {
        return std::less<const scene::SpatialField *>{}(&a, &b);
      },
      "computeValueRange",
      &scene::SpatialField::computeValueRange);

  tsd.new_usertype<scene::Array>(
      "Array",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<scene::Object>(),
      sol::meta_function::equal_to,
      [](const scene::Array &a, const scene::Array &b) { return &a == &b; },
      sol::meta_function::less_than,
      [](const scene::Array &a, const scene::Array &b) {
        return std::less<const scene::Array *>{}(&a, &b);
      },
      "elementType",
      &scene::Array::elementType,
      "size",
      &scene::Array::size,
      "elementSize",
      &scene::Array::elementSize,
      "isEmpty",
      &scene::Array::isEmpty,
      "dim",
      &scene::Array::dim,
      "setData",
      [](scene::Array &arr, sol::table data, sol::this_state s) {
        arraySetDataFromLua(arr, data, s);
      },
      "getData",
      [](scene::Array &arr, sol::this_state s) {
        return arrayGetDataAsLua(arr, s);
      });

  // Per-type Ref bindings — each lives in its own TU under bindings/refs/
  // so the nine `registerObjectMethodsOn` instantiations compile in
  // parallel.
  registerGeometryRef(tsd);
  registerMaterialRef(tsd);
  registerLightRef(tsd);
  registerCameraRef(tsd);
  registerSamplerRef(tsd);
  registerSurfaceRef(tsd);
  registerVolumeRef(tsd);
  registerSpatialFieldRef(tsd);
  registerArrayRef(tsd);
}

} // namespace tsd::scripting
