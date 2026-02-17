// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ExtensionManager.h"
// tsd_app
#include "tsd/app/Core.h"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_scripting
#ifdef TSD_USE_LUA
#include <sol/sol.hpp>
#include "tsd/scripting/LuaContext.hpp"
#endif
// std
#include <algorithm>

namespace tsd::ui::imgui {

ExtensionManager::ExtensionManager() = default;
ExtensionManager::~ExtensionManager() = default;

void ExtensionManager::initialize(tsd::app::Core *core)
{
  m_core = core;

#ifdef TSD_USE_LUA
  m_luaContext = std::make_unique<scripting::LuaContext>();
  m_luaContext->bindScene(&core->tsd.scene, "scene");

  // Route print output through tsd::core logging
  m_luaContext->setPrintCallback([](const std::string &msg) {
    std::string trimmed = msg;
    if (!trimmed.empty() && trimmed.back() == '\n')
      trimmed.pop_back();
    tsd::core::logStatus("%s", trimmed.c_str());
  });

  registerViewerBindings();
  m_luaContext->addScriptSearchPaths(getSearchPaths());
  m_menuDirty = true;
#endif
}

void ExtensionManager::refresh()
{
#ifdef TSD_USE_LUA
  clearActions();
  m_luaContext.reset();
  initialize(m_core);
#endif
}

void ExtensionManager::addMenuAction(
    const std::string &path, std::function<void()> fn)
{
  ActionEntry entry;
  entry.path = path;

  // Extract display name from the last path component
  auto pos = path.rfind('/');
  entry.displayName = (pos != std::string::npos) ? path.substr(pos + 1) : path;
  entry.fn = std::move(fn);

  m_actions.push_back(std::move(entry));
  m_menuDirty = true;
}

void ExtensionManager::addSeparator(const std::string &categoryPath)
{
  ActionEntry entry;
  entry.path = categoryPath + "/__separator__";
  entry.displayName = "__separator__";
  m_actions.push_back(std::move(entry));
  m_menuDirty = true;
}

void ExtensionManager::clearActions()
{
  m_actions.clear();
  m_menuTree.clear();
  m_menuDirty = true;
}

#ifdef TSD_USE_LUA

scripting::LuaContext &ExtensionManager::luaContext()
{
  return *m_luaContext;
}

#endif

void ExtensionManager::registerViewerBindings()
{
#ifdef TSD_USE_LUA
  auto &lua = m_luaContext->lua();
  sol::table tsd = lua["tsd"];
  sol::table viewer = lua.create_table();
  tsd["viewer"] = viewer;

  viewer["refresh"] = []() { /* no-op in viewer */ };

  viewer["addMenuAction"] = [this](const std::string &path,
                                sol::protected_function fn) {
    auto sharedFn = std::make_shared<sol::protected_function>(std::move(fn));
    addMenuAction(path, [sharedFn]() {
      auto result = (*sharedFn)();
      if (!result.valid()) {
        sol::error err = result;
        throw std::runtime_error(err.what());
      }
    });
  };

  viewer["addSeparator"] = [this](
                               const std::string &path) { addSeparator(path); };

  viewer["clearActions"] = [this]() { clearActions(); };
#endif // TSD_USE_LUA
}

const std::vector<ActionMenuNode> &ExtensionManager::getMenuTree()
{
  if (m_menuDirty) {
    rebuildMenuTree();
    m_menuDirty = false;
  }
  return m_menuTree;
}

void ExtensionManager::executeAction(size_t actionIndex)
{
  if (actionIndex >= m_actions.size())
    return;

  auto &action = m_actions[actionIndex];
  if (!action.fn)
    return;

  try {
    action.fn();
    tsd::core::logStatus("Action completed: %s", action.displayName.c_str());
  } catch (const std::exception &e) {
    tsd::core::logError(
        "Action error in '%s': %s", action.path.c_str(), e.what());
  }
}

void ExtensionManager::rebuildMenuTree()
{
  m_menuTree.clear();

  for (size_t i = 0; i < m_actions.size(); i++) {
    const auto &action = m_actions[i];
    const auto &path = action.path;

    // Split path into components
    std::vector<std::string> components;
    size_t start = 0;
    size_t end = path.find('/');

    while (end != std::string::npos) {
      components.push_back(path.substr(start, end - start));
      start = end + 1;
      end = path.find('/', start);
    }
    components.push_back(path.substr(start));

    // Navigate/create folder structure
    std::vector<ActionMenuNode> *currentLevel = &m_menuTree;
    for (size_t c = 0; c < components.size() - 1; c++) {
      const auto &folderName = components[c];

      auto it = std::find_if(currentLevel->begin(),
          currentLevel->end(),
          [&folderName](const ActionMenuNode &e) {
            return e.isFolder && e.name == folderName;
          });

      if (it == currentLevel->end()) {
        ActionMenuNode folder;
        folder.name = folderName;
        folder.isFolder = true;
        currentLevel->push_back(std::move(folder));
        it = currentLevel->end() - 1;
      }

      currentLevel = &it->children;
    }

    // Add the leaf node
    const auto &leafName = components.back();

    if (leafName == "__separator__") {
      ActionMenuNode sep;
      sep.name = "__separator__";
      sep.isSeparator = true;
      currentLevel->push_back(std::move(sep));
    } else {
      ActionMenuNode leaf;
      leaf.name = action.displayName;
      leaf.actionIndex = i;
      currentLevel->push_back(std::move(leaf));
    }
  }

  // Sort everything alphabetically (folders first, then files)
  std::function<void(std::vector<ActionMenuNode> &)> sortLevel;
  sortLevel = [&](std::vector<ActionMenuNode> &entries) {
    std::stable_sort(entries.begin(),
        entries.end(),
        [](const ActionMenuNode &a, const ActionMenuNode &b) {
          if (a.isSeparator || b.isSeparator)
            return false;
          if (a.isFolder != b.isFolder)
            return a.isFolder; // Folders first
          return a.name < b.name; // Alphabetical
        });

    for (auto &entry : entries) {
      if (entry.isFolder)
        sortLevel(entry.children);
    }
  };

  sortLevel(m_menuTree);
}

std::vector<std::string> ExtensionManager::getSearchPaths()
{
#if TSD_USE_LUA
  return scripting::LuaContext::defaultSearchPaths();
#else
  return {};
#endif
}

} // namespace tsd::ui::imgui
