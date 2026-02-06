// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#define VERBOSE_STATUS_MESSAGES 0

// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/windows/Log.h>
// tsd_io
#include <tsd/io/importers.hpp>
#include <tsd/io/procedural.hpp>
// std
#include <iostream>
#include <random>
#include <vector>

#include "DistributedViewport.h"

using TSDApplication = tsd::ui::imgui::Application;
namespace tsd_ui = tsd::ui::imgui;

namespace tsd::mpi_viewer {

///////////////////////////////////////////////////////////////////////////////
// Application definition /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

class Application : public TSDApplication
{
 public:
  Application(DistributedSceneController *controller) : m_controller(controller)
  {}
  ~Application() override = default;

  anari_viewer::WindowArray setupWindows() override
  {
    auto *log = new tsd::ui::imgui::Log(this, m_controller->appCore());

    if (!m_controller->appCore()->logging.verbose) {
      tsd::core::logStatus(
          "app window running on rank '%i'", m_controller->rank());
    }

    auto *viewport = new tsd::mpi_viewer::DistributedViewport(
        this, m_controller, "Viewport");

    anari_viewer::WindowArray windows = TSDApplication::setupWindows();
    windows.emplace_back(viewport);
    windows.emplace_back(log);

    return windows;
  }

  void uiFrameStart() override
  {
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("print ImGui ini"))
          printf("%s\n", ImGui::SaveIniSettingsToMemory());

        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("Edit")) {
        if (ImGui::MenuItem("Settings"))
          m_appSettingsDialog->show();

        ImGui::Separator();

        if (ImGui::BeginMenu("UI Layout")) {
          if (ImGui::MenuItem("Print"))
            printf("%s\n", ImGui::SaveIniSettingsToMemory());

          ImGui::Separator();

          if (ImGui::MenuItem("Reset"))
            ImGui::LoadIniSettingsFromMemory(getDefaultLayout());

          ImGui::EndMenu();
        }

        ImGui::EndMenu();
      }

      ImGui::EndMainMenuBar();
    }

    // Modals //

    if (m_appSettingsDialog->visible())
      m_appSettingsDialog->renderUI();
    if (m_taskModal->visible())
      m_taskModal->renderUI();
    if (m_fileDialog->visible())
      m_fileDialog->renderUI();
  }

  void teardown() override
  {
    m_controller->signalStop();
    m_controller->shutdown();
    TSDApplication::teardown();
  }

  const char *getDefaultLayout() const override
  {
    return R"layout(
[Window][MainDockSpace]
Pos=0,19
Size=1600,881
Collapsed=0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Viewport]
Pos=0,19
Size=1600,441
Collapsed=0
DockId=0x00000001,0

[Window][Log]
Pos=0,462
Size=1600,438
Collapsed=0
DockId=0x00000002,0

[Docking][Data]
DockSpace   ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,19 Size=1600,881 Split=Y
  DockNode  ID=0x00000001 Parent=0x80F5B4C5 SizeRef=1600,441 CentralNode=1 Selected=0xC450F867
  DockNode  ID=0x00000002 Parent=0x80F5B4C5 SizeRef=1600,438 Selected=0x139FDA3F
)layout";
  }

 private:
  DistributedSceneController *m_controller{nullptr};
};

} // namespace tsd::mpi_viewer

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv)
{
  tsd::core::setLogToStdout();
  {
    auto controller =
        std::make_unique<tsd::mpi_viewer::DistributedSceneController>();

    controller->initialize(argc, argv);

    if (controller->isMain()) {
      auto app =
          std::make_unique<tsd::mpi_viewer::Application>(controller.get());
      app->run(1920, 1080, "Distributed TSD Viewer");
    } else {
      do {
        controller->executeFrame();
      } while (controller->isRunning());
      controller->shutdown();
    }
  }

  return 0;
}
