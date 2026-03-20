// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/scene/Object.hpp"
#include "tsd/scene/objects/Array.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

anari::DataType arrayTypeFromString(const std::string &typeStr);

scene::Object *extractObjectPtr(sol::object luaObj);

void arraySetDataFromLua(scene::Array &arr, sol::table data, sol::this_state s);

void arraySetObjectsFromLua(scene::Array &arr, sol::table data);

void inferArrayDimsFromLuaData(sol::table data,
    anari::DataType elemType,
    size_t &items0,
    size_t &items1,
    size_t &items2);

scene::ArrayRef setParameterArrayFromLua(scene::Object &obj,
    const std::string &name,
    const std::string &typeStr,
    sol::table data,
    sol::this_state s);

scene::ArrayRef setParameterArrayFromLua(scene::Object &obj,
    const std::string &name,
    const std::string &typeStr,
    size_t items0,
    size_t items1,
    size_t items2,
    sol::table data,
    sol::this_state s);

} // namespace tsd::scripting
