// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/scripting/LuaContext.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scripting/LuaBindings.hpp"

#include <sol/sol.hpp>

#include <fmt/format.h>

#include <filesystem>

namespace tsd::scripting {

struct LuaContext::Impl
{
  sol::state lua;
  core::Scene *boundScene{nullptr};
  std::unique_ptr<core::Scene> ownedScene;
  PrintCallback printCallback;
  std::string outputBuffer;
};

LuaContext::LuaContext() : m_impl(std::make_unique<Impl>())
{
  m_impl->lua.open_libraries(sol::lib::base,
      sol::lib::package,
      sol::lib::coroutine,
      sol::lib::string,
      sol::lib::os,
      sol::lib::math,
      sol::lib::table,
      sol::lib::io,
      sol::lib::utf8);

  m_impl->lua.set_function("print", [this](sol::variadic_args va) {
    fmt::memory_buffer buf;
    bool first = true;
    for (auto arg : va) {
      if (!first)
        buf.push_back('\t');
      first = false;

      sol::object obj = arg;
      if (obj.is<std::string>()) {
        auto s = obj.as<std::string>();
        buf.append(s.data(), s.data() + s.size());
      } else if (obj.is<double>()) {
        fmt::format_to(std::back_inserter(buf), "{}", obj.as<double>());
      } else if (obj.is<bool>()) {
        auto s = obj.as<bool>() ? std::string_view("true")
                                : std::string_view("false");
        buf.append(s.data(), s.data() + s.size());
      } else if (obj.is<sol::lua_nil_t>()) {
        constexpr std::string_view nil = "nil";
        buf.append(nil.data(), nil.data() + nil.size());
      } else {
        auto s = sol::type_name(m_impl->lua, obj.get_type());
        buf.append(s.data(), s.data() + s.size());
      }
    }
    buf.push_back('\n');

    auto output = fmt::to_string(buf);
    m_impl->outputBuffer += output;

    if (m_impl->printCallback) {
      m_impl->printCallback(output);
    }
  });

  registerAllBindings(m_impl->lua);
}

LuaContext::~LuaContext() = default;

ExecutionResult LuaContext::executeFile(const std::string &filepath)
{
  ExecutionResult result;
  m_impl->outputBuffer.clear();

  try {
    auto loadResult =
        m_impl->lua.safe_script_file(filepath, sol::script_pass_on_error);

    if (!loadResult.valid()) {
      sol::error err = loadResult;
      result.success = false;
      result.error = err.what();
    } else {
      result.success = true;
    }
  } catch (const std::exception &e) {
    result.success = false;
    result.error = e.what();
  }

  result.output = m_impl->outputBuffer;
  return result;
}

ExecutionResult LuaContext::executeString(const std::string &script)
{
  ExecutionResult result;
  m_impl->outputBuffer.clear();

  try {
    auto loadResult =
        m_impl->lua.safe_script(script, sol::script_pass_on_error);

    if (!loadResult.valid()) {
      sol::error err = loadResult;
      result.success = false;
      result.error = err.what();
    } else {
      result.success = true;
    }
  } catch (const std::exception &e) {
    result.success = false;
    result.error = e.what();
  }

  result.output = m_impl->outputBuffer;
  return result;
}

std::vector<std::string> LuaContext::addScriptSearchPaths(
    const std::vector<std::string> &paths)
{
  namespace fs = std::filesystem;
  std::vector<std::string> errors;

  for (const auto &path : paths) {
    std::string current = m_impl->lua["package"]["path"];
    m_impl->lua["package"]["path"] =
        current + ";" + path + "/?.lua;" + path + "/?/init.lua";
  }

  for (const auto &path : paths) {
    fs::path initFile = fs::path(path) / "init.lua";
    if (fs::exists(initFile) && fs::is_regular_file(initFile)) {
      auto result = m_impl->lua.safe_script_file(
          initFile.string(), sol::script_pass_on_error);
      if (!result.valid()) {
        sol::error err = result;
        std::string msg =
            fmt::format("Error in {}: {}", initFile.string(), err.what());
        tsd::core::logWarning("%s", msg.c_str());
        errors.push_back(std::move(msg));
      }
    }
  }

  return errors;
}

void LuaContext::bindScene(core::Scene *scene, const std::string &varName)
{
  m_impl->boundScene = scene;
  m_impl->ownedScene.reset();
  if (scene) {
    m_impl->lua[varName] = scene;
  } else {
    m_impl->lua[varName] = sol::lua_nil;
  }
}

core::Scene *LuaContext::createOwnedScene(const std::string &varName)
{
  m_impl->ownedScene = std::make_unique<core::Scene>();
  m_impl->boundScene = m_impl->ownedScene.get();
  m_impl->lua[varName] = m_impl->boundScene;
  return m_impl->boundScene;
}

core::Scene *LuaContext::boundScene() const
{
  return m_impl->boundScene;
}

void LuaContext::setPrintCallback(PrintCallback callback)
{
  m_impl->printCallback = std::move(callback);
}

sol::state &LuaContext::lua()
{
  return m_impl->lua;
}

const sol::state &LuaContext::lua() const
{
  return m_impl->lua;
}

std::vector<std::string> LuaContext::defaultSearchPaths()
{
  namespace fs = std::filesystem;
  std::vector<std::string> paths;

  // Search-priority order: first entry = searched first.

  // Source tree (development only)
#ifdef TSD_SOURCE_DIR
  try {
    fs::path sourcePath = TSD_SOURCE_DIR;
    fs::path scriptsPath = sourcePath / "tsd" / "scripts";
    if (fs::exists(scriptsPath) && fs::is_directory(scriptsPath))
      paths.push_back(scriptsPath.string());
  } catch (const std::exception &) {
  }
#endif

  // Installation directory
  try {
    fs::path exePath = fs::current_path();
    fs::path installPath = exePath / ".." / "share" / "tsd" / "scripts";
    installPath = fs::canonical(installPath);
    if (fs::exists(installPath) && fs::is_directory(installPath))
      paths.push_back(installPath.string());
  } catch (const std::exception &) {
  }

  // User config directory
#ifdef _WIN32
  const char *appData = std::getenv("APPDATA");
  if (appData) {
    fs::path userPath = fs::path(appData) / "tsd" / "scripts";
    if (fs::exists(userPath) && fs::is_directory(userPath))
      paths.push_back(userPath.string());
  }
#else
  const char *home = std::getenv("HOME");
  if (home) {
    fs::path userPath = fs::path(home) / ".config" / "tsd" / "scripts";
    if (fs::exists(userPath) && fs::is_directory(userPath))
      paths.push_back(userPath.string());
  }
#endif

  // TSD_LUA_PACKAGE_PATHS environment variable
  const char *envPath = std::getenv("TSD_LUA_PACKAGE_PATHS");
  if (envPath) {
    std::string pathStr(envPath);
#ifdef _WIN32
    const char pathSep = ';';
#else
    const char pathSep = ':';
#endif

    size_t start = 0;
    size_t end = pathStr.find(pathSep);
    while (end != std::string::npos) {
      paths.push_back(pathStr.substr(start, end - start));
      start = end + 1;
      end = pathStr.find(pathSep, start);
    }
    paths.push_back(pathStr.substr(start));
  }

  return paths;
}

} // namespace tsd::scripting
