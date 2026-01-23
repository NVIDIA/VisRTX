// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Terminal.h"
#include "Application.h"
// imgui
#include <imgui.h>
// std
#include <algorithm>

#ifdef TSD_USE_LUA
#include "tsd/scripting/LuaContext.hpp"
#include "tsd/ui/imgui/ExtensionManager.h"
#endif

namespace tsd::ui::imgui {

Terminal::Terminal(Application *app) : Window(app, "Terminal")
{
#ifdef TSD_USE_LUA
  if (auto *sm = m_app->extensionManager()) {
    m_luaContext = &sm->luaContext();

    m_luaContext->setPrintCallback(
        [this](const std::string &msg) { addOutput(msg); });
  }

  addOutput("TSD Lua Terminal\n");
  addOutput("Type 'help' for usage hints.\n");
  addOutput("'scene' is bound to the current TSD scene.\n\n");
#else
  addOutput("Lua scripting is not enabled.\n");
  addOutput("Build with -DTSD_USE_LUA=ON to enable.\n");
#endif
}

Terminal::~Terminal() = default;

void Terminal::buildUI()
{
  // Compute input widget size based on number of lines, clamped to a reasonable
  // max. We want the input box to grow with the number of lines, but not exceed
  // a certain height.
  int lines = 1;
  for (int i = 0; m_inputBuffer[i]; i++)
    if (m_inputBuffer[i] == '\n')
      lines++;
  lines = std::clamp(lines, 1, 10);
  float inputHeight =
      ImGui::GetTextLineHeight() * lines + ImGui::GetStyle().FramePadding.y * 2;
  float footerHeight = ImGui::GetStyle().ItemSpacing.y + 1
      + ImGui::GetStyle().ItemSpacing.y + inputHeight;

  // Outer child owns the scrollbar; InputTextMultiline is sized to its
  // content so it has no internal scroll of its own.
  ImGui::BeginChild("OutputScroll",
      ImVec2(-FLT_MIN, -footerHeight),
      ImGuiChildFlags_None,
      ImGuiWindowFlags_HorizontalScrollbar);

  float outputHeight = ImGui::GetTextLineHeight() * m_outputLineCount
      + ImGui::GetStyle().FramePadding.y * 2;

  ImGui::InputTextMultiline("##Output",
      const_cast<char *>(m_outputText.c_str()),
      m_outputText.size() + 1,
      ImVec2(-FLT_MIN, outputHeight),
      ImGuiInputTextFlags_ReadOnly);

  if (m_scrollToBottom) {
    ImGui::SetScrollHereY(1.0f);
    m_scrollToBottom = false;
  }

  ImGui::EndChild();

  ImGui::Separator();

  ImGuiInputTextFlags inputFlags = ImGuiInputTextFlags_EnterReturnsTrue
      | ImGuiInputTextFlags_CtrlEnterForNewLine
      | ImGuiInputTextFlags_EscapeClearsAll
      | ImGuiInputTextFlags_CallbackAlways;

  bool reclaimFocus = false;
  if (ImGui::InputTextMultiline("##Input",
          m_inputBuffer,
          IM_ARRAYSIZE(m_inputBuffer),
          ImVec2(-FLT_MIN, inputHeight),
          inputFlags,
          &Terminal::inputCallback,
          this)) {
    if (m_inputBuffer[0]) {
      executeCommand(m_inputBuffer);
      m_inputBuffer[0] = '\0';
    }
    reclaimFocus = true;
  }

  if (!m_inputBuffer[0]) {
    ImVec2 pos = ImGui::GetItemRectMin();
    const auto &style = ImGui::GetStyle();
    pos.x += style.FramePadding.x;
    pos.y += style.FramePadding.y;
    ImGui::GetWindowDrawList()->AddText(pos,
        ImGui::GetColorU32(ImGuiCol_TextDisabled),
        "Enter to submit, Ctrl+Enter for newline, Alt+Up/Down for history");
  }

  ImGui::SetItemDefaultFocus();
  if (reclaimFocus)
    ImGui::SetKeyboardFocusHere(-1);
}

void Terminal::addOutput(const std::string &text)
{
  for (char c : text)
    if (c == '\n')
      m_outputLineCount++;
  m_outputText += text;
  m_scrollToBottom = true;
}

void Terminal::executeCommand(const std::string &command)
{
  m_history.push_back(command);
  if (m_history.size() > m_maxHistorySize) {
    m_history.pop_front();
  }
  m_historyPos = -1;

  addOutput("> " + command + "\n");

#ifdef TSD_USE_LUA
  if (command == "clear") {
    clear();
    return;
  }
  if (command == "help") {
    addOutput(
        "Available globals:\n"
        "  scene     - The current TSD scene\n"
        "  tsd       - The TSD Lua module\n"
        "\n"
        "TSD namespaces:\n"
        "  tsd.io       - Importers and procedural generators\n"
        "  tsd.render   - Rendering functions\n"
        "  tsd.viewer   - Viewer integration (refresh, etc.)\n"
        "\n"
        "Built-in commands:\n"
        "  clear     - Clear the terminal\n"
        "  help      - Show this help\n"
        "\n"
        "Example:\n"
        "  tsd.io.generateRandomSpheres(scene)\n"
        "  tsd.viewer.refresh()\n");
    return;
  }

  auto result = m_luaContext->executeString(command);
  if (!result.success) {
    addOutput("Error: " + result.error + "\n");
  }
#else
  addOutput("Lua scripting not available.\n");
#endif
}

void Terminal::clear()
{
  m_outputText.clear();
  m_outputLineCount = 1;
}

int Terminal::inputCallback(ImGuiInputTextCallbackData *data)
{
  Terminal *t = (Terminal *)data->UserData;

  if (data->EventFlag == ImGuiInputTextFlags_CallbackAlways) {
    bool altUp = ImGui::GetIO().KeyAlt && ImGui::IsKeyPressed(ImGuiKey_UpArrow);
    bool altDown =
        ImGui::GetIO().KeyAlt && ImGui::IsKeyPressed(ImGuiKey_DownArrow);

    if (altUp || altDown) {
      const int prevPos = t->m_historyPos;

      if (altUp) {
        if (t->m_historyPos == -1)
          t->m_historyPos = (int)t->m_history.size() - 1;
        else if (t->m_historyPos > 0)
          t->m_historyPos--;
      } else {
        if (t->m_historyPos != -1) {
          if (++t->m_historyPos >= (int)t->m_history.size())
            t->m_historyPos = -1;
        }
      }

      if (prevPos != t->m_historyPos) {
        const char *str =
            (t->m_historyPos >= 0) ? t->m_history[t->m_historyPos].c_str() : "";
        data->DeleteChars(0, data->BufTextLen);
        data->InsertChars(0, str);
      }
    }
  }

  return 0;
}

} // namespace tsd::ui::imgui
