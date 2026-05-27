// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Registers the base `scene::Object` usertype. Sits in its own TU because
// the `registerObjectMethodsOn` instantiation for `scene::Object` is one
// of the two heaviest sol2 templates in the bindings (the other nine
// instantiations are the per-type Refs under refs/).

#include "ObjectMethodBindings.hpp"
#include "ObjectUsertype.hpp"
#include "tsd/scene/Object.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerObjectUsertype(sol::table &tsd)
{
  auto objectType = tsd.new_usertype<scene::Object>(
      "Object", sol::no_constructor, "index", &scene::Object::index);

  registerObjectMethodsOn(
      objectType, [](scene::Object &o) -> scene::Object * { return &o; });
}

} // namespace tsd::scripting
