// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"
// std
#include <deque>
#include <string>

#ifdef TSD_USE_LUA
namespace tsd::scripting {
class LuaContext;
}
#endif

namespace tsd::ui::imgui {

struct Terminal : public Window
{
  Terminal(Application *app);
  ~Terminal();

  void buildUI() override;

  void addOutput(const std::string &text);
  void executeCommand(const std::string &command);

 private:
  void clear();
  static int inputCallback(ImGuiInputTextCallbackData *data);

  // Data //

  char m_inputBuffer[1024]{};
  std::deque<std::string> m_history;
  int m_historyPos{-1};
  size_t m_maxHistorySize{100};

  std::string m_outputText;
  int m_outputLineCount{1};
  bool m_scrollToBottom{false};

#ifdef TSD_USE_LUA
  tsd::scripting::LuaContext *m_luaContext{nullptr};
#endif
};

} // namespace tsd::ui::imgui
