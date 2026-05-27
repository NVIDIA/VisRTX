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
// Each binding is a one-line trampoline that forwards to a free function
// in ObjectMethodImpls.cpp. The method body code is emitted exactly once
// for the whole TU instead of once per UserType, so per-type
// instantiation cost is dominated by sol2's call-wrapping machinery
// rather than by the optimizer chewing through duplicated bodies.
//
// Null-handling policy (implemented inside the free functions):
//   - Mutating operations (setParameter, setParameterArray): throw on null
//   - Read operations (getParameter, name, type, etc.): return nil/default
//   - Idempotent remove operations (removeParameter, etc.): silent no-op

#include "ObjectMethodImpls.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

template <typename UserType, typename Accessor>
void registerObjectMethodsOn(sol::usertype<UserType> &ut, Accessor access)
{
  ut["name"] = sol::property(
      [access](UserType &u) { return objectGetName(access(u)); },
      [access](UserType &u, const std::string &n) {
        objectSetName(access(u), n);
      });

  ut["subtype"] = [access](UserType &u) { return objectGetSubtype(access(u)); };
  ut["type"] = [access](UserType &u) { return objectGetType(access(u)); };

  ut["setParameter"] = [access](UserType &u,
                            const std::string &name,
                            sol::object value,
                            sol::this_state s) {
    objectSetParameter(access(u), name, value, s);
  };

  ut["setParameterArray"] = sol::overload(
      [access](UserType &u,
          const std::string &name,
          const std::string &typeStr,
          sol::table data,
          sol::this_state s) {
        return objectSetParameterArray0(access(u), name, typeStr, data, s);
      },
      [access](UserType &u,
          const std::string &name,
          const std::string &typeStr,
          size_t items0,
          sol::table data,
          sol::this_state s) {
        return objectSetParameterArray1(
            access(u), name, typeStr, items0, data, s);
      },
      [access](UserType &u,
          const std::string &name,
          const std::string &typeStr,
          size_t items0,
          size_t items1,
          sol::table data,
          sol::this_state s) {
        return objectSetParameterArray2(
            access(u), name, typeStr, items0, items1, data, s);
      },
      [access](UserType &u,
          const std::string &name,
          const std::string &typeStr,
          size_t items0,
          size_t items1,
          size_t items2,
          sol::table data,
          sol::this_state s) {
        return objectSetParameterArray3(
            access(u), name, typeStr, items0, items1, items2, data, s);
      });

  ut["getParameter"] =
      [access](UserType &u, const std::string &name, sol::this_state s) {
        return objectGetParameter(access(u), name, s);
      };

  ut["parameter"] = [access](UserType &u, const std::string &name) {
    return objectGetParameterPtr(access(u), name);
  };

  ut["removeParameter"] = [access](UserType &u, const std::string &name) {
    objectRemoveParameter(access(u), name);
  };

  ut["removeAllParameters"] = [access](UserType &u) {
    objectRemoveAllParameters(access(u));
  };

  ut["numParameters"] = [access](UserType &u) {
    return objectNumParameters(access(u));
  };

  ut["parameterNameAt"] = [access](UserType &u, size_t i) {
    return objectParameterNameAt(access(u), i);
  };

  ut["setMetadata"] =
      [access](UserType &u, const std::string &key, sol::object value) {
        objectSetMetadata(access(u), key, value);
      };

  ut["getMetadata"] =
      [access](UserType &u, const std::string &key, sol::this_state s) {
        return objectGetMetadata(access(u), key, s);
      };

  ut["removeMetadata"] = [access](UserType &u, const std::string &key) {
    objectRemoveMetadata(access(u), key);
  };

  ut["numMetadata"] = [access](UserType &u) {
    return objectNumMetadata(access(u));
  };

  ut["getMetadataName"] = [access](UserType &u, size_t i) {
    return objectGetMetadataName(access(u), i);
  };
}

} // namespace tsd::scripting
