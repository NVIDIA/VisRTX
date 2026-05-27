// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Free-function implementations of every method that
// `registerObjectMethodsOn` binds onto a sol2 usertype.
//
// All methods take `scene::Object *` (possibly null) and handle the
// null case internally per the policy documented in
// ObjectMethodBindings.hpp. This keeps the template trampoline in
// that header trivial — body code is emitted exactly once in
// ObjectMethodImpls.cpp, not per UserType.

#include "tsd/scene/objects/Array.hpp"

#include <sol/sol.hpp>

#include <anari/anari_cpp.hpp>

#include <string>

namespace tsd::scene {
struct Object;
class Parameter;
} // namespace tsd::scene

namespace tsd::scripting {

std::string objectGetName(scene::Object *obj);
void objectSetName(scene::Object *obj, const std::string &n);
std::string objectGetSubtype(scene::Object *obj);
anari::DataType objectGetType(scene::Object *obj);

void objectSetParameter(scene::Object *obj,
    const std::string &name,
    sol::object value,
    sol::this_state s);

scene::ArrayRef objectSetParameterArray0(scene::Object *obj,
    const std::string &name,
    const std::string &typeStr,
    sol::table data,
    sol::this_state s);
scene::ArrayRef objectSetParameterArray1(scene::Object *obj,
    const std::string &name,
    const std::string &typeStr,
    size_t items0,
    sol::table data,
    sol::this_state s);
scene::ArrayRef objectSetParameterArray2(scene::Object *obj,
    const std::string &name,
    const std::string &typeStr,
    size_t items0,
    size_t items1,
    sol::table data,
    sol::this_state s);
scene::ArrayRef objectSetParameterArray3(scene::Object *obj,
    const std::string &name,
    const std::string &typeStr,
    size_t items0,
    size_t items1,
    size_t items2,
    sol::table data,
    sol::this_state s);

sol::object objectGetParameter(
    scene::Object *obj, const std::string &name, sol::this_state s);
const scene::Parameter *objectGetParameterPtr(
    scene::Object *obj, const std::string &name);
void objectRemoveParameter(scene::Object *obj, const std::string &name);
void objectRemoveAllParameters(scene::Object *obj);
size_t objectNumParameters(scene::Object *obj);
const char *objectParameterNameAt(scene::Object *obj, size_t i);

void objectSetMetadata(
    scene::Object *obj, const std::string &key, sol::object value);
sol::object objectGetMetadata(
    scene::Object *obj, const std::string &key, sol::this_state s);
void objectRemoveMetadata(scene::Object *obj, const std::string &key);
size_t objectNumMetadata(scene::Object *obj);
const char *objectGetMetadataName(scene::Object *obj, size_t i);

} // namespace tsd::scripting
