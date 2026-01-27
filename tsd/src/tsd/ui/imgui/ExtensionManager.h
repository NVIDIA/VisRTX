// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#ifdef TSD_USE_LUA
namespace tsd::scripting {
class LuaContext;
}
#endif

namespace tsd {
namespace app {
struct Core;
}
namespace ui::imgui {

struct ActionEntry
{
  std::string path;        // "glTF/Geometry/Box"
  std::string displayName; // "Box"
  std::function<void()> fn;
};

struct ActionMenuNode
{
  std::string name;
  bool isFolder{false};
  bool isSeparator{false};
  std::vector<ActionMenuNode> children;
  size_t actionIndex{SIZE_MAX}; // index into m_actions, SIZE_MAX = folder
};

class ExtensionManager
{
 public:
  ExtensionManager();
  ~ExtensionManager();

  void initialize(tsd::app::Core *core);
  void refresh();

  void addMenuAction(const std::string &path, std::function<void()> fn);
  void addSeparator(const std::string &categoryPath);
  void clearActions();

  const std::vector<ActionMenuNode> &getMenuTree();
  void executeAction(size_t actionIndex);

#ifdef TSD_USE_LUA
  scripting::LuaContext &luaContext();
#endif

  static std::vector<std::string> getSearchPaths();

 private:
#ifdef TSD_USE_LUA
  void registerViewerBindings();
#endif
  void rebuildMenuTree();

#ifdef TSD_USE_LUA
  std::unique_ptr<scripting::LuaContext> m_luaContext;
#endif
  std::vector<ActionEntry> m_actions;
  std::vector<ActionMenuNode> m_menuTree;
  bool m_menuDirty{true};
  tsd::app::Core *m_core{nullptr};
};

} // namespace ui::imgui
} // namespace tsd
