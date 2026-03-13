// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sol/sol.hpp>

namespace tsd::scene {
struct Object;
}

namespace tsd::scripting {

// Helper to set parameter on an Object from a Lua value.
// Supports: bool, int, float, string, float2/3/4, mat4, ArrayRef, SamplerRef,
// GeometryRef, MaterialRef, SpatialFieldRef,
// and Lua tables of 2-4 numbers (auto-converted to float2/3/4).
void setParameterFromLua(
    scene::Object *obj, const std::string &name, sol::object value);

// Helper to get parameter value from an Object as a Lua object.
// Returns sol::nil if the parameter doesn't exist or has an unsupported type.
sol::object getParameterAsLua(
    sol::state_view lua, const scene::Object *obj, const std::string &name);

// Apply all string-keyed entries from a Lua table as parameters on an Object.
void applyParameterTable(scene::Object *obj, const sol::table &params);

// Helper to set a metadata value on an Object from a Lua value.
// Supports: bool, int, float, string.
void setMetadataFromLua(
    scene::Object *obj, const std::string &key, sol::object value);

// Helper to get a metadata value from an Object as a Lua object.
// Returns sol::nil if the key doesn't exist or has an unsupported type.
sol::object getMetadataAsLua(
    sol::state_view lua, const scene::Object *obj, const std::string &key);

} // namespace tsd::scripting
