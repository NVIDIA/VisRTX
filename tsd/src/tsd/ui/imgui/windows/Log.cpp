// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Log.h"
// tsd_app
#include "tsd/app/Core.h"
// std
#include <cstdio>
// fmt
#include <fmt/format.h>

namespace tsd::ui::imgui {

Log::Log(Application *app, bool installAsLoggingTarget)
    : Window(app, "Log"), m_isLoggingTarget(installAsLoggingTarget)
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
    tsd::core::setLoggingCallback(
        [window = this](tsd::core::LogLevel level, std::string msg) {
          window->addText(level, msg);
        });
  }
}

Log::~Log()
{
  if (m_isLoggingTarget)
    tsd::core::setLogToStdout();
}

void Log::buildUI()
{
  if (ImGui::BeginPopup("Options")) {
    ImGui::Checkbox("Log verbose ANARI messages", &appCore()->logging.verbose);
    ImGui::Checkbox("Echo log to stdout", &appCore()->logging.echoOutput);
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

void Log::addText(tsd::core::LogLevel level, const std::string &msg)
{
  m_colorIDs.push_back(static_cast<int>(level));

  if (appCore() && appCore()->logging.echoOutput)
    fmt::print(stdout, "{}", msg);

  auto old_size = m_buf.size();
  m_buf.append(msg.c_str());

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

} // namespace tsd::ui::imgui
