// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../AppCore.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"
// std
#include <array>

namespace tsd_viewer {

struct Log : public anari_viewer::windows::Window
{
  Log(AppCore *ctx, bool installAsLoggingTarget = true);
  ~Log();

  void buildUI() override;

 private:
  void addText(tsd::LogLevel level, const char *fmt, va_list &args);
  void showLine(int line_no, bool useFilter);
  void clear();

  // Data //

  AppCore *m_core{nullptr};
  bool m_isLoggingTarget{false};

  ImGuiTextBuffer m_buf;
  ImGuiTextFilter m_filter;
  ImVector<int> m_lineOffsets;
  ImVector<int> m_colorIDs;

  std::array<ImVec4, 7> m_colors;

  bool m_autoScroll{true};
};

} // namespace tsd_viewer
