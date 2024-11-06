// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AppContext.h"
#include "Logging.h"
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
#include <iostream>
#include <vector>

using tsd_viewer::ImporterType;

static tsd_viewer::AppContext *g_context = nullptr;
static const char *g_defaultLayout =
    R"layout(
[Window][MainDockSpace]
Pos=0,25
Size=1920,1054
Collapsed=0

[Window][Viewport]
Pos=550,25
Size=684,806
Collapsed=0
DockId=0x00000003,0

[Window][Secondary View]
Pos=1236,25
Size=684,806
Collapsed=0
DockId=0x00000004,0

[Window][Log]
Pos=550,833
Size=1370,246
Collapsed=0
DockId=0x00000002,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Object Tree]
Pos=0,25
Size=548,347
Collapsed=0
DockId=0x00000007,0

[Window][Object Editor]
Pos=0,374
Size=548,705
Collapsed=0
DockId=0x00000008,0

[Table][0x39E9F5ED,1]
Column 0  Weight=1.0000

[Table][0x418F6C9E,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xE57DC2D0,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x65B57849,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Docking][Data]
DockSpace       ID=0x782A6D6B Window=0xDEDC5B90 Pos=0,25 Size=1920,1054 Split=X Selected=0x13926F0B
  DockNode      ID=0x00000005 Parent=0x782A6D6B SizeRef=548,626 Split=Y Selected=0x1FD98235
    DockNode    ID=0x00000007 Parent=0x00000005 SizeRef=549,288 Selected=0x1FD98235
    DockNode    ID=0x00000008 Parent=0x00000005 SizeRef=549,584 Selected=0xAFC1D085
  DockNode      ID=0x00000006 Parent=0x782A6D6B SizeRef=1370,626 Split=Y Selected=0x13926F0B
    DockNode    ID=0x00000001 Parent=0x00000006 SizeRef=1049,626 Split=X Selected=0x13926F0B
      DockNode  ID=0x00000003 Parent=0x00000001 SizeRef=684,626 CentralNode=1 Selected=0x13926F0B
      DockNode  ID=0x00000004 Parent=0x00000001 SizeRef=684,626 Selected=0xBAF13E1E
    DockNode    ID=0x00000002 Parent=0x00000006 SizeRef=1049,246 Selected=0x64F50EE5
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

    auto *manipulator = &m_manipulator;

    auto *log = new Log(g_context);
    m_viewport = new Viewport(g_context, manipulator, "Viewport");
    m_viewport2 = new Viewport(g_context, manipulator, "Secondary View");
    m_viewport2->hide();
    auto *oeditor = new ObjectEditor(g_context);
    auto *otree = new ObjectTree(g_context);

    anari_viewer::WindowArray windows;
    windows.emplace_back(m_viewport);
    windows.emplace_back(m_viewport2);
    windows.emplace_back(oeditor);
    windows.emplace_back(otree);
    windows.emplace_back(log);

    // Populate scene //

    m_sceneLoadFuture = std::async([viewport = m_viewport, m = manipulator]() {
      tsd::generate_material_orb(g_context->tsd.ctx);

      g_context->setupSceneFromCommandLine(true);

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

      logging::logStatus("...scene load complete!");
      logging::logStatus(
          "%s", tsd::objectDBInfo(g_context->tsd.ctx.objectDB()).c_str());
      g_context->tsd.sceneLoadComplete = true;

      viewport->setLibrary(g_context->commandLine.libraryList[0], false);

      m->setConfig(
          tsd::float3(0.f, 0.136f, 0.f), 0.75f, tsd::float2(330.f, 35.f));
    });

    return windows;
  }

  void uiFrameStart() override
  {
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("print ImGui ini")) {
          const char *info = ImGui::SaveIniSettingsToMemory();
          printf("%s\n", info);
        }

        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("View")) {
        ImGui::Checkbox("secondary view", m_viewport2->visiblePtr());
        ImGui::EndMenu();
      }

      ImGui::EndMainMenuBar();
    }
  }

  void teardown() override
  {
    if (m_sceneLoadFuture.valid())
      m_sceneLoadFuture.get();
    anari_viewer::ui::shutdown();
  }

 private:
  manipulators::Orbit m_manipulator;
  tsd_viewer::Viewport *m_viewport{nullptr};
  tsd_viewer::Viewport *m_viewport2{nullptr};
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
    app.run(1920, 1080, "TSD Material Explorer");
    g_context = nullptr;
  }

  return 0;
}
