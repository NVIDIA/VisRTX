// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/scripting/LuaBindings.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerAllBindings(sol::state &lua)
{
  sol::table tsd = lua.create_named_table("tsd");
  tsd["io"] = lua.create_table();
  tsd["render"] = lua.create_table();

  // Register bindings in order of dependency
  registerMathBindings(lua);
  registerCoreBindings(lua);
  registerObjectBindings(lua);
  registerLayerBindings(lua);
  registerIOBindings(lua);
  registerRenderBindings(lua);
}

} // namespace tsd::scripting
