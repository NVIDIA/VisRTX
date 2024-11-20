// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "BaseApplication.h"
// tsd
#include "tsd_ui.h"
// anari_viewer
#include "anari_viewer/ui_anari.h"

namespace tsd_viewer {

BaseApplication::BaseApplication(int argc, const char **argv)
{
  appCore()->parseCommandLine(argc, argv);
}

BaseApplication::~BaseApplication() = default;

AppCore *BaseApplication::appCore()
{
  return &m_core;
}

anari_viewer::WindowArray BaseApplication::setupWindows()
{
  anari_viewer::ui::init();

  ImGuiIO &io = ImGui::GetIO();
  io.FontGlobalScale = 1.5f;
  io.IniFilename = nullptr;

  if (appCore()->commandLine.useDefaultLayout)
    ImGui::LoadIniSettingsFromMemory(getDefaultLayout());

  m_appSettings = std::make_unique<tsd_viewer::AppSettings>();

  return {};
}

void BaseApplication::uiFrameStart()
{
  // Handle app shortcuts //

  ImGuiIO &io = ImGui::GetIO();
  if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S))
    saveContext();

  // Main Menu //

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Save", "CTRL+S"))
        saveContext();

      ImGui::Separator();

      if (ImGui::MenuItem("Print UI Layout")) {
        const char *info = ImGui::SaveIniSettingsToMemory();
        printf("%s\n", info);
      }

      ImGui::Separator();

      if (ImGui::MenuItem("Quit", "SHIFT+Q"))
        std::exit(0);

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Edit")) {
      if (ImGui::MenuItem("Settings"))
        m_appSettings->show();
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
      for (auto &w : m_windows) {
        ImGui::PushID(&w);
        ImGui::Checkbox(w->name(), w->visiblePtr());
        ImGui::PopID();
      }
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();

    // Modals //

    if (m_appSettings->visible())
      m_appSettings->renderUI();
  }
}

void BaseApplication::teardown()
{
  anari_viewer::ui::shutdown();
}

void BaseApplication::saveContext()
{
  tsd::save_Context(appCore()->tsd.ctx, "state.tsd");
  tsd::logStatus("context saved to 'state.tsd'");
}

void BaseApplication::setWindowArray(const anari_viewer::WindowArray &wa)
{
  for (auto &w : wa)
    m_windows.push_back(w.get());
}

} // namespace tsd_viewer