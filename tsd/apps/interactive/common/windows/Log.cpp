// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Log.h"
// std
#include <cstdio>

namespace tsd_viewer {

// Helper functions ///////////////////////////////////////////////////////////

static void printMessage(
    tsd::LogLevel level, const char *fmt, va_list &args, bool workOnCopy)
{
  ImGuiTextBuffer buf;
  if (workOnCopy) {
    va_list args_copy;
    va_copy(args_copy, args);
    buf.appendfv(fmt, args_copy);
    va_end(args_copy);
  } else {
    buf.appendfv(fmt, args);
  }
  std::string msg(buf.begin(), buf.begin() + buf.size());
  printf("%s\n", msg.c_str());
}

// Log definitions ////////////////////////////////////////////////////////////

Log::Log(AppCore *core, bool installAsLoggingTarget)
    : anari_viewer::windows::Window(core->application, "Log", true),
      m_core(core),
      m_isLoggingTarget(installAsLoggingTarget)
{
  this->clear();

  m_colors[0] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // Status
  m_colors[1] = ImVec4(1.0f, 0.0f, 0.0f, 1.0f); // Error
  m_colors[2] = ImVec4(1.0f, 0.5f, 0.0f, 1.0f); // Warning
  m_colors[3] = ImVec4(1.0f, 1.0f, 0.5f, 1.0f); // Perf
  m_colors[4] = ImVec4(0.7f, 0.7f, 1.0f, 1.0f); // Info
  m_colors[5] = ImVec4(0.4f, 0.4f, 1.0f, 1.0f); // Debug
  m_colors[6] = ImVec4(1.0f, 0.0f, 1.0f, 1.0f); // Log window problem

  if (installAsLoggingTarget) {
    tsd::setLoggingCallback(
        [window = this](tsd::LogLevel level, const char *fmt, va_list &args) {
          window->addText(level, fmt, args);
        });
  }
}

Log::~Log()
{
  if (m_isLoggingTarget) {
    tsd::setLoggingCallback(
        [](tsd::LogLevel level, const char *fmt, va_list &args) {
          printMessage(level, fmt, args, false);
        });
  }
}

void Log::buildUI()
{
  if (ImGui::BeginPopup("Options")) {
    ImGui::Checkbox("Log verbose ANARI messages", &m_core->logging.verbose);
    ImGui::Checkbox("Echo log to stdout", &m_core->logging.echoOutput);
    ImGui::Checkbox("Auto-scroll", &m_autoScroll);
    ImGui::EndPopup();
  }

  if (ImGui::Button("Options"))
    ImGui::OpenPopup("Options");

  ImGui::SameLine();

  if (ImGui::Button("Clear Log"))
    this->clear();

  ImGui::SameLine();

  m_filter.Draw("Filter", -100.0f);

  ImGui::Separator();

  ImGui::BeginChild(
      "scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

  if (m_filter.IsActive()) {
    for (int line_no = 0; line_no < m_lineOffsets.Size; line_no++)
      showLine(line_no, true);
  } else {
    ImGuiListClipper clipper;
    clipper.Begin(m_lineOffsets.Size);
    while (clipper.Step()) {
      for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd;
          line_no++)
        showLine(line_no, false);
    }
    clipper.End();
  }
  ImGui::PopStyleVar();

  if (m_autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
    ImGui::SetScrollHereY(1.0f);

  ImGui::EndChild();
}

void Log::addText(tsd::LogLevel level, const char *fmt, va_list &args)
{
  m_colorIDs.push_back(static_cast<int>(level));

  if (m_core && m_core->logging.echoOutput)
    printMessage(level, fmt, args, true);

  auto old_size = m_buf.size();
  m_buf.appendfv(fmt, args);
  m_buf.append("\n");

  for (int new_size = m_buf.size(); old_size < new_size; old_size++) {
    if (m_buf[old_size] == '\n') {
      m_lineOffsets.push_back(old_size + 1);
      if (old_size + 1 < new_size)
        m_colorIDs.push_back(m_colorIDs.back());
    }
  }
}

void Log::showLine(int line_no, bool useFilter)
{
  const char *buf = m_buf.begin();
  const char *buf_end = m_buf.end();

  const char *line_start = buf + m_lineOffsets[line_no];
  const char *line_end = (line_no + 1 < m_lineOffsets.Size)
      ? (buf + m_lineOffsets[line_no + 1] - 1)
      : buf_end;

  bool setColor = line_start != line_end;

  if (setColor) {
    auto id = std::clamp(m_colorIDs[line_no], 0, int(m_colors.size() - 1));
    ImGui::PushStyleColor(ImGuiCol_Text, m_colors.at(id));
  }

  if (!useFilter || m_filter.PassFilter(line_start, line_end))
    ImGui::TextUnformatted(line_start, line_end);

  if (setColor)
    ImGui::PopStyleColor();
}

void Log::clear()
{
  m_buf.clear();
  m_lineOffsets.clear();
  m_lineOffsets.push_back(0);
  m_colorIDs.clear();
}

} // namespace tsd_viewer
