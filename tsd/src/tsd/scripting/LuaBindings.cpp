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

  // Terminal command registry: scripts populate `tsd.terminal.commands` with
  // { run = fn(args) -> string, summary = string } records; the interactive
  // frontends dispatch a line's first token here before evaluating as Lua.
  sol::table terminal = lua.create_table();
  terminal["commands"] = lua.create_table();
  // Default help describing the C++-provided exposure. Owned here so it is
  // available even without the script pack; the fallback prints it and the
  // script-registered `help` command embeds it in its overview.
  terminal["defaultHelp"] =
      "Available globals:\n"
      "  scene         The current TSD scene\n"
      "  animationMgr  Animation collection + time/frame control\n"
      "  tsd           The TSD Lua module\n"
      "\n"
      "TSD namespaces:\n"
      "  tsd.io      Importers and procedural generators\n"
      "  tsd.render  Offline rendering (loadDevice, createRenderIndex, ...)\n"
      "  tsd.viewer  Viewer integration (refresh, addMenuAction; viewer only)\n"
      "\n"
      "Example:\n"
      "  tsd.io.generateRandomSpheres(scene)\n"
      "  print(scene:numberOfObjects(tsd.GEOMETRY))\n";
  tsd["terminal"] = terminal;

  // Register bindings in order of dependency
  registerMathBindings(lua);
  registerContextBindings(lua);
  registerAnimationManagerBindings(lua);
  registerObjectBindings(lua);
  registerLayerBindings(lua);
  registerIOBindings(lua);
  registerRenderBindings(lua);
}

} // namespace tsd::scripting
