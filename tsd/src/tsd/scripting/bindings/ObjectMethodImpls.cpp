// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ObjectMethodImpls.hpp"

#include "ArrayHelpers.hpp"
#include "ParameterHelpers.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/scene/Object.hpp"
#include "tsd/scene/Parameter.hpp"
#include "tsd/scene/objects/Array.hpp"

#include <sol/sol.hpp>

#include <stdexcept>

namespace tsd::scripting {

// Null-handling policy:
//   - Mutating operations (setParameter, setParameterArray, set*): throw
//     when obj is null.
//   - Read operations (get*, num*, name*): return empty/default values.
//   - Idempotent remove operations: silent no-op when obj is null.

namespace {

[[noreturn]] void throwInvalidRef()
{
  throw std::runtime_error("attempt to set parameter on invalid reference");
}

} // namespace

std::string objectGetName(scene::Object *obj)
{
  return obj ? std::string(obj->name()) : std::string();
}

void objectSetName(scene::Object *obj, const std::string &n)
{
  if (obj)
    obj->setName(n.c_str());
}

std::string objectGetSubtype(scene::Object *obj)
{
  return obj ? obj->subtype().str() : std::string();
}

anari::DataType objectGetType(scene::Object *obj)
{
  return obj ? obj->type() : ANARI_UNKNOWN;
}

void objectSetParameter(scene::Object *obj,
    const std::string &name,
    sol::object value,
    sol::this_state /*s*/)
{
  if (!obj)
    throwInvalidRef();
  setParameterFromLua(obj, name, value);
}

scene::ArrayRef objectSetParameterArray0(scene::Object *obj,
    const std::string &name,
    const std::string &typeStr,
    sol::table data,
    sol::this_state s)
{
  if (!obj)
    throwInvalidRef();
  return setParameterArrayFromLua(*obj, name, typeStr, data, s);
}

scene::ArrayRef objectSetParameterArray1(scene::Object *obj,
    const std::string &name,
    const std::string &typeStr,
    size_t items0,
    sol::table data,
    sol::this_state s)
{
  if (!obj)
    throwInvalidRef();
  return setParameterArrayFromLua(*obj, name, typeStr, items0, 0, 0, data, s);
}

scene::ArrayRef objectSetParameterArray2(scene::Object *obj,
    const std::string &name,
    const std::string &typeStr,
    size_t items0,
    size_t items1,
    sol::table data,
    sol::this_state s)
{
  if (!obj)
    throwInvalidRef();
  return setParameterArrayFromLua(
      *obj, name, typeStr, items0, items1, 0, data, s);
}

scene::ArrayRef objectSetParameterArray3(scene::Object *obj,
    const std::string &name,
    const std::string &typeStr,
    size_t items0,
    size_t items1,
    size_t items2,
    sol::table data,
    sol::this_state s)
{
  if (!obj)
    throwInvalidRef();
  return setParameterArrayFromLua(
      *obj, name, typeStr, items0, items1, items2, data, s);
}

sol::object objectGetParameter(
    scene::Object *obj, const std::string &name, sol::this_state s)
{
  if (!obj)
    return sol::lua_nil;
  return getParameterAsLua(sol::state_view(s), obj, name);
}

const scene::Parameter *objectGetParameterPtr(
    scene::Object *obj, const std::string &name)
{
  return obj ? obj->parameter(core::Token(name)) : nullptr;
}

void objectRemoveParameter(scene::Object *obj, const std::string &name)
{
  if (obj)
    obj->removeParameter(core::Token(name));
}

void objectRemoveAllParameters(scene::Object *obj)
{
  if (obj)
    obj->removeAllParameters();
}

size_t objectNumParameters(scene::Object *obj)
{
  return obj ? obj->numParameters() : 0;
}

const char *objectParameterNameAt(scene::Object *obj, size_t i)
{
  return obj ? obj->parameterNameAt(i) : "";
}

void objectSetMetadata(
    scene::Object *obj, const std::string &key, sol::object value)
{
  if (obj)
    setMetadataFromLua(obj, key, value);
}

sol::object objectGetMetadata(
    scene::Object *obj, const std::string &key, sol::this_state s)
{
  if (!obj)
    return sol::lua_nil;
  return getMetadataAsLua(sol::state_view(s), obj, key);
}

void objectRemoveMetadata(scene::Object *obj, const std::string &key)
{
  if (obj)
    obj->removeMetadata(key);
}

size_t objectNumMetadata(scene::Object *obj)
{
  return obj ? obj->numMetadata() : 0;
}

const char *objectGetMetadataName(scene::Object *obj, size_t i)
{
  if (!obj)
    return "";
  const char *n = obj->getMetadataName(i);
  return n ? n : "";
}

} // namespace tsd::scripting
