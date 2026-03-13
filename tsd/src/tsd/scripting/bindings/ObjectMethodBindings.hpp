// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared template that registers all Object methods on a sol::usertype,
// parameterized by an accessor that extracts the scene::Object* from the
// bound user type.  Used for both the Object type itself and for
// ObjectPoolRef<T> types, so the method list is defined in exactly one place.
//
// Accessor signature:  scene::Object *(UserType &u)
//   - For Object:  returns &u  (never null)
//   - For Ref:     returns r.data() when valid, nullptr otherwise
//
// Null-handling policy (follows Lua conventions):
//   - Mutating operations (setParameter, setParameterArray): throw on null
//   - Read operations (getParameter, name, type, etc.): return nil/default
//   - Idempotent remove operations (removeParameter, etc.): silent no-op

#include "ArrayHelpers.hpp"
#include "ParameterHelpers.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/scene/Parameter.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

template <typename UserType, typename Accessor>
void registerObjectMethodsOn(sol::usertype<UserType> &ut, Accessor access)
{
  ut["name"] = sol::property(
      [access](UserType &u) -> std::string {
        if (auto *obj = access(u))
          return obj->name();
        return "";
      },
      [access](UserType &u, const std::string &n) {
        if (auto *obj = access(u))
          obj->setName(n.c_str());
      });

  ut["subtype"] = [access](UserType &u) -> std::string {
    if (auto *obj = access(u))
      return obj->subtype().str();
    return "";
  };

  ut["type"] = [access](UserType &u) -> ANARIDataType {
    if (auto *obj = access(u))
      return obj->type();
    return ANARI_UNKNOWN;
  };

  ut["setParameter"] = [access](UserType &u,
                            const std::string &name,
                            sol::object value,
                            sol::this_state s) {
    auto *obj = access(u);
    if (!obj)
      throw std::runtime_error(
          "attempt to set parameter on invalid reference");
    setParameterFromLua(obj, name, value);
  };

  ut["setParameterArray"] = sol::overload(
      [access](UserType &u,
          const std::string &name,
          const std::string &typeStr,
          sol::table data,
          sol::this_state s) {
        auto *obj = access(u);
        if (!obj)
          throw std::runtime_error(
              "attempt to set parameter on invalid reference");
        return setParameterArrayFromLua(*obj, name, typeStr, data, s);
      },
      [access](UserType &u,
          const std::string &name,
          const std::string &typeStr,
          size_t items0,
          sol::table data,
          sol::this_state s) {
        auto *obj = access(u);
        if (!obj)
          throw std::runtime_error(
              "attempt to set parameter on invalid reference");
        return setParameterArrayFromLua(
            *obj, name, typeStr, items0, 0, 0, data, s);
      },
      [access](UserType &u,
          const std::string &name,
          const std::string &typeStr,
          size_t items0,
          size_t items1,
          sol::table data,
          sol::this_state s) {
        auto *obj = access(u);
        if (!obj)
          throw std::runtime_error(
              "attempt to set parameter on invalid reference");
        return setParameterArrayFromLua(
            *obj, name, typeStr, items0, items1, 0, data, s);
      },
      [access](UserType &u,
          const std::string &name,
          const std::string &typeStr,
          size_t items0,
          size_t items1,
          size_t items2,
          sol::table data,
          sol::this_state s) {
        auto *obj = access(u);
        if (!obj)
          throw std::runtime_error(
              "attempt to set parameter on invalid reference");
        return setParameterArrayFromLua(
            *obj, name, typeStr, items0, items1, items2, data, s);
      });

  ut["getParameter"] = [access](UserType &u,
                            const std::string &name,
                            sol::this_state s) -> sol::object {
    auto *obj = access(u);
    if (!obj)
      return sol::lua_nil;
    return getParameterAsLua(sol::state_view(s), obj, name);
  };

  ut["parameter"] = [access](UserType &u,
                        const std::string &name) -> const scene::Parameter * {
    auto *obj = access(u);
    if (!obj)
      return nullptr;
    return obj->parameter(core::Token(name));
  };

  ut["removeParameter"] = [access](UserType &u, const std::string &name) {
    if (auto *obj = access(u))
      obj->removeParameter(core::Token(name));
  };

  ut["removeAllParameters"] = [access](UserType &u) {
    if (auto *obj = access(u))
      obj->removeAllParameters();
  };

  ut["numParameters"] = [access](UserType &u) -> size_t {
    if (auto *obj = access(u))
      return obj->numParameters();
    return 0;
  };

  ut["parameterNameAt"] = [access](UserType &u, size_t i) -> const char * {
    if (auto *obj = access(u))
      return obj->parameterNameAt(i);
    return "";
  };

  // Metadata
  ut["setMetadata"] =
      [access](UserType &u, const std::string &key, sol::object value) {
        if (auto *obj = access(u))
          setMetadataFromLua(obj, key, value);
      };

  ut["getMetadata"] = [access](UserType &u,
                          const std::string &key,
                          sol::this_state s) -> sol::object {
    auto *obj = access(u);
    if (!obj)
      return sol::lua_nil;
    return getMetadataAsLua(sol::state_view(s), obj, key);
  };

  ut["removeMetadata"] = [access](UserType &u, const std::string &key) {
    if (auto *obj = access(u))
      obj->removeMetadata(key);
  };

  ut["numMetadata"] = [access](UserType &u) -> size_t {
    if (auto *obj = access(u))
      return obj->numMetadata();
    return 0;
  };

  ut["getMetadataName"] = [access](UserType &u, size_t i) -> const char * {
    if (auto *obj = access(u)) {
      const char *n = obj->getMetadataName(i);
      return n ? n : "";
    }
    return "";
  };
}

} // namespace tsd::scripting
