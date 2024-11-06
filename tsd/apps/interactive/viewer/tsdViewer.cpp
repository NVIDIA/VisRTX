// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AppContext.h"
#include "Logging.h"
#include "modals/AppSettings.h"
#include "windows/DatabaseEditor.h"
#include "windows/Log.h"
#include "windows/ObjectEditor.h"
#include "windows/ObjectTree.h"
#include "windows/Viewport.h"
// anari_viewer
#include "anari_viewer/Application.h"
// tsd
#include "tsd/TSD.hpp"
#include "tsd_ui.h"
// std
#include <chrono>
#include <iostream>
#include <vector>

using tsd_viewer::ImporterType;

static tsd_viewer::AppContext *g_context = nullptr;
static const char *g_defaultLayout =
    R"layout(
[Window][MainDockSpace]
Pos=0,25
Size=1600,874
Collapsed=0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Viewport]
Pos=551,25
Size=525,617
Collapsed=0
DockId=0x00000007,0

[Window][Database Editor]
Pos=0,461
Size=549,438
Collapsed=0
DockId=0x00000004,1

[Window][Object Tree]
Pos=0,25
Size=549,434
Collapsed=0
DockId=0x00000003,0

[Window][Object Editor]
Pos=0,461
Size=549,438
Collapsed=0
DockId=0x00000004,0

[Window][Log]
Pos=551,644
Size=1049,255
Collapsed=0
DockId=0x00000006,0

[Window][Secondary View]
Pos=1078,25
Size=522,617
Collapsed=0
DockId=0x00000008,0

[Table][0x44C159D3,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x9E1800B1,1]
Column 0  Weight=1.0000

[Docking][Data]
DockSpace       ID=0x782A6D6B Window=0xDEDC5B90 Pos=0,25 Size=1600,874 Split=X
  DockNode      ID=0x00000001 Parent=0x782A6D6B SizeRef=549,1079 Split=Y Selected=0x3912ED61
    DockNode    ID=0x00000003 Parent=0x00000001 SizeRef=549,434 Selected=0xEA0B185E
    DockNode    ID=0x00000004 Parent=0x00000001 SizeRef=549,438 Selected=0xAFC1D085
  DockNode      ID=0x00000002 Parent=0x782A6D6B SizeRef=1369,1079 Split=Y Selected=0x13926F0B
    DockNode    ID=0x00000005 Parent=0x00000002 SizeRef=1049,617 Split=X Selected=0x13926F0B
      DockNode  ID=0x00000007 Parent=0x00000005 SizeRef=525,617 CentralNode=1 Selected=0x13926F0B
      DockNode  ID=0x00000008 Parent=0x00000005 SizeRef=522,617 Selected=0xBAF13E1E
    DockNode    ID=0x00000006 Parent=0x00000002 SizeRef=1049,255 Selected=0x64F50EE5
)layout";

namespace tsd_viewer {

class Application : public anari_viewer::Application
{
 public:
  Application() = default;
  ~Application() override = default;

  anari_viewer::WindowArray setupWindows() override
  {
    anari_viewer::ui::init();

    // ImGui //

    ImGuiIO &io = ImGui::GetIO();
    io.FontGlobalScale = 1.5f;
    io.IniFilename = nullptr;

    if (g_context->commandLine.useDefaultLayout)
      ImGui::LoadIniSettingsFromMemory(g_defaultLayout);

    m_appSettings = std::make_unique<tsd_viewer::AppSettings>();

    auto *log = new Log(g_context);
    m_viewport = new Viewport(g_context, &m_manipulator, "Viewport");
    m_viewport2 = new Viewport(g_context, &m_manipulator, "Secondary View");
    m_viewport2->hide();
    auto *dbeditor = new DatabaseEditor(g_context);
    auto *oeditor = new ObjectEditor(g_context);
    auto *otree = new ObjectTree(g_context);

    anari_viewer::WindowArray windows;
    windows.emplace_back(m_viewport);
    windows.emplace_back(m_viewport2);
    windows.emplace_back(dbeditor);
    windows.emplace_back(oeditor);
    windows.emplace_back(otree);
    windows.emplace_back(log);

    // Populate scene //

    m_sceneLoadFuture = std::async([viewport = m_viewport]() {
      auto loadStart = std::chrono::steady_clock::now();
      g_context->setupSceneFromCommandLine();
      auto loadEnd = std::chrono::steady_clock::now();
      auto loadSeconds =
          std::chrono::duration<float>(loadEnd - loadStart).count();

      if (!g_context->commandLine.loadingContext) {
        logging::logStatus("...setting up directional light");

        auto light = g_context->tsd.ctx.createObject<tsd::Light>(
            tsd::tokens::light::directional);
        light->setName("mainLight");
        light->setParameter("direction", tsd::float2(0.f, 240.f));

        g_context->tsd.ctx.tree.insert_first_child(
            g_context->tsd.ctx.tree.root(),
            tsd::utility::Any(ANARI_LIGHT, light.index()));
      }

      logging::logStatus("...scene load complete! (%.3fs)", loadSeconds);
      logging::logStatus(
          "%s", tsd::objectDBInfo(g_context->tsd.ctx.objectDB()).c_str());
      g_context->tsd.sceneLoadComplete = true;

      viewport->setLibrary(g_context->commandLine.libraryList[0], false);
    });

    return windows;
  }

  void uiFrameStart() override
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
        ImGui::Checkbox("secondary view", m_viewport2->visiblePtr());
        ImGui::EndMenu();
      }

      ImGui::EndMainMenuBar();
    }

    // Modals //

    if (m_appSettings->visible())
      m_appSettings->renderUI();
  }

  void teardown() override
  {
    if (m_sceneLoadFuture.valid())
      m_sceneLoadFuture.get();
    anari_viewer::ui::shutdown();
  }

 private:
  void saveContext()
  {
    tsd::save_Context(g_context->tsd.ctx, "state.tsd");
    logging::logStatus("context saved to 'state.tsd'");
  }

  manipulators::Orbit m_manipulator;
  tsd_viewer::Viewport *m_viewport{nullptr};
  tsd_viewer::Viewport *m_viewport2{nullptr};
  std::unique_ptr<tsd_viewer::AppSettings> m_appSettings;
  std::future<void> m_sceneLoadFuture;
}; // namespace tsd_viewer

} // namespace tsd_viewer

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
  {
    auto context = std::make_unique<tsd_viewer::AppContext>();
    g_context = context.get();

    context->parseCommandLine(argc, argv);

    tsd_viewer::Application app;
    app.run(1920, 1080, "TSD Viewer");
    g_context = nullptr;
  }

  return 0;
}
