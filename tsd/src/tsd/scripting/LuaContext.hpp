// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sol {
class state;
}

namespace tsd::scene {
struct Scene;
}

namespace tsd::animation {
struct AnimationManager;
}

namespace tsd::scripting {

using PrintCallback = std::function<void(const std::string &)>;

struct ExecutionResult
{
  bool success{false};
  std::string error;
  std::string output;
};

// Result of attempting to dispatch a console line to a registered terminal
// command (the `tsd.terminal.commands` table). `handled` is false when the
// line's first token is not a registered command, in which case the caller
// should evaluate the line as Lua instead.
struct ConsoleCommandResult
{
  bool handled{false};
  bool success{false};
  std::string output; // string returned by the command's run()
  std::string error;
};

class LuaContext
{
 public:
  LuaContext();
  ~LuaContext();

  LuaContext(const LuaContext &) = delete;
  LuaContext &operator=(const LuaContext &) = delete;
  LuaContext(LuaContext &&) = delete;
  LuaContext &operator=(LuaContext &&) = delete;

  ExecutionResult executeFile(const std::string &filepath);
  ExecutionResult executeString(const std::string &script);

  // If `line`'s first whitespace-delimited token names a registered command in
  // `tsd.terminal.commands`, call its `run(args)` with the remaining tokens and
  // return the result. Single-line input only; returns `handled == false`
  // otherwise so the caller can evaluate `line` as Lua.
  ConsoleCommandResult runRegisteredCommand(const std::string &line);

  // The C++-owned default help text (`tsd.terminal.defaultHelp`), shown by the
  // frontends when no `help` command is registered.
  std::string consoleDefaultHelp();

  // Scene is NOT owned by LuaContext
  void bindScene(scene::Scene *scene, const std::string &varName = "scene");

  // Scene IS owned by LuaContext
  scene::Scene *createOwnedScene(const std::string &varName = "scene");

  void bindAnimationManager(tsd::animation::AnimationManager *sa,
      const std::string &varName = "animationMgr");

  scene::Scene *boundScene() const;

  // Adds paths to Lua's package.path and executes any init.lua found in them.
  // Returns errors encountered (empty on success).
  std::vector<std::string> addScriptSearchPaths(
      const std::vector<std::string> &paths);

  // Returns search paths in priority order:
  //   1. <source>/tsd/scripts/     (dev builds with TSD_SOURCE_DIR)
  //   2. <exe>/../share/tsd/scripts/
  //   3. ~/.config/tsd/scripts/    (or %APPDATA%/tsd/scripts/ on Windows)
  //   4. TSD_LUA_PACKAGE_PATHS env var (: or ; separated)
  static std::vector<std::string> defaultSearchPaths();

  void setPrintCallback(PrintCallback callback);

  sol::state &lua();
  const sol::state &lua() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace tsd::scripting
