// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AppContext.h"
#include "modals/AppSettings.h"
#include "windows/IsosurfaceEditor.h"
#include "windows/Log.h"
#include "windows/ObjectEditor.h"
#include "windows/ObjectTree.h"
#include "windows/TransferFunctionEditor.h"
#include "windows/Viewport.h"
// anari_viewer
#include "anari_viewer/Application.h"
// tsd
#include "tsd/TSD.hpp"
#include "tsd_ui.h"
// std
#include <algorithm>
#include <iostream>
#include <vector>

using tsd_viewer::ImporterType;

static std::string g_filename;
static tsd_viewer::AppContext *g_context = nullptr;
static const char *g_defaultLayout =
    R"layout(
[Window][MainDockSpace]
Pos=0,22
Size=2489,1211
Collapsed=0

[Window][Viewport]
Pos=550,22
Size=1389,963
Collapsed=0
DockId=0x00000003,0

[Window][Secondary View]
Pos=1236,25
Size=684,806
Collapsed=0
DockId=0x00000004,0

[Window][Log]
Pos=550,987
Size=1389,246
Collapsed=0
DockId=0x00000002,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Object Tree]
Pos=0,22
Size=548,450
Collapsed=0
DockId=0x00000007,0

[Window][Object Editor]
Pos=0,474
Size=548,759
Collapsed=0
DockId=0x00000008,0

[Window][TF Editor]
Pos=1941,22
Size=548,606
Collapsed=0
DockId=0x00000009,0

[Window][Isosurface Editor]
Pos=1941,630
Size=548,603
Collapsed=0
DockId=0x0000000A,0

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

[Table][0x3EF92DF9,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xB8840F96,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xD3B898DD,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x348E3E86,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x4D7E8B1D,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x2F5F88FA,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Docking][Data]
DockSpace         ID=0x782A6D6B Window=0xDEDC5B90 Pos=0,22 Size=2489,1211 Split=X Selected=0x13926F0B
  DockNode        ID=0x0000000B Parent=0x782A6D6B SizeRef=1326,1475 Split=X
    DockNode      ID=0x00000005 Parent=0x0000000B SizeRef=548,626 Split=Y Selected=0x1FD98235
      DockNode    ID=0x00000007 Parent=0x00000005 SizeRef=548,162 Selected=0x1FD98235
      DockNode    ID=0x00000008 Parent=0x00000005 SizeRef=548,273 Selected=0xAFC1D085
    DockNode      ID=0x00000006 Parent=0x0000000B SizeRef=1370,626 Split=Y Selected=0x13926F0B
      DockNode    ID=0x00000001 Parent=0x00000006 SizeRef=1049,626 Split=X Selected=0x13926F0B
        DockNode  ID=0x00000003 Parent=0x00000001 SizeRef=684,626 CentralNode=1 Selected=0x13926F0B
        DockNode  ID=0x00000004 Parent=0x00000001 SizeRef=684,626 Selected=0xBAF13E1E
      DockNode    ID=0x00000002 Parent=0x00000006 SizeRef=1049,246 Selected=0x64F50EE5
  DockNode        ID=0x0000000C Parent=0x782A6D6B SizeRef=548,1475 Split=Y Selected=0xE3280322
    DockNode      ID=0x00000009 Parent=0x0000000C SizeRef=548,738 Selected=0xE3280322
    DockNode      ID=0x0000000A Parent=0x0000000C SizeRef=548,735 Selected=0x2468BDAC
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

    auto *manipulator = &m_manipulator;

    auto *log = new Log(g_context);
    m_viewport = new Viewport(g_context, manipulator, "Viewport");
    m_viewport2 = new Viewport(g_context, manipulator, "Secondary View");
    m_viewport2->hide();
    auto *oeditor = new ObjectEditor(g_context);
    auto *otree = new ObjectTree(g_context);
    auto *tfeditor = new TransferFunctionEditor(g_context);
    auto *isoeditor = new IsosurfaceEditor(g_context);

    anari_viewer::WindowArray windows;
    windows.emplace_back(m_viewport);
    windows.emplace_back(m_viewport2);
    windows.emplace_back(oeditor);
    windows.emplace_back(otree);
    windows.emplace_back(log);
    windows.emplace_back(tfeditor);
    windows.emplace_back(isoeditor);

    // Populate scene //

    m_sceneLoadFuture = std::async([viewport = m_viewport,
                                       m = manipulator,
                                       tf = tfeditor]() {
      auto colorArray = g_context->tsd.ctx.createArray(ANARI_FLOAT32_VEC3, 256);
      auto opacityArray = g_context->tsd.ctx.createArray(ANARI_FLOAT32, 256);

      tsd::VolumeRef volume;

      if (!g_filename.empty()) {
        volume = tsd::import_volume(
            g_context->tsd.ctx, g_filename.c_str(), colorArray, opacityArray);
      } else {
        volume = tsd::generate_noiseVolume(
            g_context->tsd.ctx, colorArray, opacityArray);
      }

      if (volume) {
        g_context->tsd.selectedObject = volume.data();
        tf->setValueRange(volume->parameter("valueRange")
                              ->value()
                              .getAs<tsd::float2>(ANARI_FLOAT32_BOX1));
      }

      g_context->setupSceneFromCommandLine(true);

      if (!g_context->commandLine.loadingContext) {
        tsd::logStatus("...setting up directional light");

        auto light = g_context->tsd.ctx.createObject<tsd::Light>(
            tsd::tokens::light::directional);
        light->setName("mainLight");
        light->setParameter("direction", tsd::float2(0.f, 240.f));

        g_context->tsd.ctx.tree.insert_first_child(
            g_context->tsd.ctx.tree.root(),
            tsd::utility::Any(ANARI_LIGHT, light.index()));
      }

      tsd::logStatus("...scene load complete!");
      tsd::logStatus(
          "%s", tsd::objectDBInfo(g_context->tsd.ctx.objectDB()).c_str());
      g_context->tsd.sceneLoadComplete = true;

      viewport->setLibrary(g_context->commandLine.libraryList[0], false);

      tf->setUpdateCallback([=](const tsd::float2 &valueRange,
                                const std::vector<tsd::float4> &co) mutable {
        auto *colors = colorArray->mapAs<tsd::float3>();
        auto *opacities = opacityArray->mapAs<float>();
        std::transform(co.begin(), co.end(), colors, [](const tsd::float4 &v) {
          return tsd::float3(v.x, v.y, v.z);
        });
        std::transform(
            co.begin(), co.end(), opacities, [](const tsd::float4 &v) {
              return v.w;
            });
        colorArray->unmap();
        opacityArray->unmap();

        volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
      });
    });

    return windows;
  }

  void uiFrameStart() override
  {
    // Main Menu //

    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("print ImGui ini")) {
          const char *info = ImGui::SaveIniSettingsToMemory();
          printf("%s\n", info);
        }

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
  manipulators::Orbit m_manipulator;
  tsd::Array *m_colorArray{nullptr};
  tsd::Array *m_opacityArray{nullptr};
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
    if (argc > 1)
      g_filename = argv[1];

    auto context = std::make_unique<tsd_viewer::AppContext>();
    g_context = context.get();

    context->parseCommandLine(argc, argv);

    tsd_viewer::Application app;
    app.run(1920, 1080, "TSD Volume Viewer");
    g_context = nullptr;
  }

  return 0;
}
