// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared helper: registers the common ObjectPoolRef<T> usertype
// (constructor, validity, index, to_string) and forwards all Object
// methods via ObjectMethodBindings. Each per-type TU instantiates this
// for exactly one T and then layers on type-specific accessors.

#include "ObjectMethodBindings.hpp"
#include "tsd/scene/Object.hpp"

#include <fmt/format.h>
#include <sol/sol.hpp>

namespace tsd::scripting {

template <typename T>
auto registerObjectPoolRef(sol::table &tsd, const char *name)
{
  using Ref = scene::ObjectPoolRef<T>;
  auto refType = tsd.new_usertype<Ref>(
      name,
      sol::no_constructor,
      "valid",
      &Ref::valid,
      "index",
      [](const Ref &r) -> size_t { return r.index(); },
      sol::meta_function::to_string,
      [name](const Ref &r) {
        if (!r.valid())
          return fmt::format("{}(invalid)", name);
        return fmt::format("{}({})", name, r.index());
      });

  registerObjectMethodsOn(refType,
      [](Ref &r) -> scene::Object * { return r.valid() ? r.data() : nullptr; });

  return refType;
}

} // namespace tsd::scripting
