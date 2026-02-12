// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/windows/CameraPoses.h>
#include <tsd/ui/imgui/windows/DatabaseEditor.h>
#include <tsd/ui/imgui/windows/IsosurfaceEditor.h>
#include <tsd/ui/imgui/windows/LayerTree.h>
#include <tsd/ui/imgui/windows/Log.h>
#include <tsd/ui/imgui/windows/ObjectEditor.h>
#include <tsd/ui/imgui/windows/TransferFunctionEditor.h>
#include <tsd/ui/imgui/windows/Viewport.h>
// std
#include <chrono>

#include "NodeEditor.h"
#include "NodeInfoWindow.h"

namespace tsd::demo {

using TSDApplication = tsd::ui::imgui::Application;
namespace tsd_ui = tsd::ui::imgui;

class Application : public TSDApplication
{
 public:
  Application(int argc, const char *argv[]) : TSDApplication(argc, argv) {}
  ~Application() override = default;

  anari_viewer::WindowArray setupWindows() override
  {
    ImNodes::CreateContext();

    auto windows = TSDApplication::setupWindows();

    auto *core = appCore();

    auto *cameras = new tsd_ui::CameraPoses(this);
    auto *log = new tsd_ui::Log(this);
    m_viewport =
        new tsd_ui::Viewport(this, &core->view.manipulator, "Viewport");
    m_viewport->setDeviceChangeCb([&](const std::string &libName) {
      auto &adm = appCore()->anari;
      auto &scene = appCore()->tsd.scene;
      // Use the same ANARI device for the graph as we are in the viewport
      m_graph.setANARIDevice(adm.loadDevice(libName));
      if (!libName.empty()) {
        tsd::core::logStatus(
            "[viskores] graph now using ANARI library '%s'", libName.c_str());
      }
    });
    auto *dbeditor = new tsd_ui::DatabaseEditor(this);
    auto *oeditor = new tsd_ui::ObjectEditor(this);
    auto *otree = new tsd_ui::LayerTree(this);
    auto *tfeditor = new tsd_ui::TransferFunctionEditor(this);
    auto *isoeditor = new tsd_ui::IsosurfaceEditor(this);

    auto ninfo = new tsd::viskores_graph::NodeInfoWindow(this);
    m_neditor = new tsd::viskores_graph::NodeEditor(this, &m_graph, ninfo);

    windows.emplace_back(cameras);
    windows.emplace_back(m_viewport);
    windows.emplace_back(dbeditor);
    windows.emplace_back(oeditor);
    windows.emplace_back(otree);
    windows.emplace_back(log);
    windows.emplace_back(tfeditor);
    windows.emplace_back(ninfo);
    windows.emplace_back(m_neditor);

    setWindowArray(windows);

    tfeditor->hide();

    // Populate scene //

    auto populateScene = [vp = m_viewport, core = core]() {
      auto &scene = core->tsd.scene;

      const bool setupDefaultLight = !core->commandLine.loadedFromStateFile
          && scene.numberOfObjects(ANARI_LIGHT) == 0;
      if (setupDefaultLight) {
        tsd::core::logStatus("...setting up default light");

        auto light = scene.createObject<tsd::core::Light>(
            tsd::core::tokens::light::directional);
        light->setName("mainLight");
        light->setParameter("direction", tsd::math::float2(0.f, 240.f));

        scene.defaultLayer()->root()->insert_first_child({light});
      }

      core->tsd.sceneLoadComplete = true;

      vp->setLibraryToDefault();
    };

#if 1
    showTaskModal(populateScene, "Please Wait: Loading Scene...");
#else
    populateScene();
#endif

    return windows;
  }

  void uiFrameEnd() override
  {
    m_graph.update(viskores::graph::GraphExecutionPolicy::ALL_ASYNC, [&]() {
      auto &instances = m_graph.getANARIInstances();
      m_viewport->setExternalInstances(instances.data(), instances.size());
      m_neditor->updateNodeSummary();
    });
    TSDApplication::uiFrameEnd();
  }

  void teardown() override
  {
    m_graph.sync();
    ImNodes::DestroyContext();
    TSDApplication::teardown();
  }

  const char *getDefaultLayout() const override
  {
    return R"layout(
[Window][MainDockSpace]
Pos=0,26
Size=1920,1105
Collapsed=0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Viewport]
Pos=432,26
Size=1488,528
Collapsed=0
DockId=0x0000000F,0

[Window][Database Editor]
Pos=0,399
Size=430,300
Collapsed=0
DockId=0x0000000D,1

[Window][Layers]
Pos=0,26
Size=430,371
Collapsed=0
DockId=0x00000008,0

[Window][Object Editor]
Pos=0,399
Size=430,300
Collapsed=0
DockId=0x0000000D,0

[Window][Log]
Pos=432,889
Size=1488,242
Collapsed=0
DockId=0x00000005,0

[Window][Secondary View]
Pos=1237,26
Size=683,848
Collapsed=0
DockId=0x00000007,0

[Window][Isosurface Editor]
Pos=1370,26
Size=550,1054
Collapsed=0
DockId=0x0000000C,0

[Window][TF Editor]
Pos=1370,26
Size=550,590
Collapsed=0
DockId=0x0000000B,0

[Window][Camera Poses]
Pos=0,26
Size=430,371
Collapsed=0
DockId=0x00000008,1

[Window][Node Info]
Pos=0,701
Size=430,430
Collapsed=0
DockId=0x0000000E,0

[Window][Node Editor]
Pos=432,556
Size=1488,331
Collapsed=0
DockId=0x00000010,0

[Window][##]
Pos=792,507
Size=336,116
Collapsed=0

[Table][0x44C159D3,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x9E1800B1,1]
Column 0  Weight=1.0000

[Table][0xFAE9835A,1]
Column 0  Weight=1.0000

[Table][0x413D162D,1]
Column 0  Weight=1.0000

[Table][0x34853C34,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xEEE697AB,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x50507568,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xF4075185,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Docking][Data]
DockSpace           ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1105 Split=X
  DockNode          ID=0x00000003 Parent=0x80F5B4C5 SizeRef=1368,1054 Split=X
    DockNode        ID=0x00000001 Parent=0x00000003 SizeRef=430,1105 Split=Y Selected=0xCD8384B1
      DockNode      ID=0x00000008 Parent=0x00000001 SizeRef=547,354 Selected=0xCD8384B1
      DockNode      ID=0x00000009 Parent=0x00000001 SizeRef=547,698 Split=Y Selected=0x82B4C496
        DockNode    ID=0x0000000D Parent=0x00000009 SizeRef=547,286 Selected=0x82B4C496
        DockNode    ID=0x0000000E Parent=0x00000009 SizeRef=547,410 Selected=0x7ECBF265
    DockNode        ID=0x00000002 Parent=0x00000003 SizeRef=1488,1105 Split=Y
      DockNode      ID=0x00000004 Parent=0x00000002 SizeRef=1370,861 Split=X Selected=0xC450F867
        DockNode    ID=0x00000006 Parent=0x00000004 SizeRef=685,848 Split=Y Selected=0xC450F867
          DockNode  ID=0x0000000F Parent=0x00000006 SizeRef=1371,528 CentralNode=1 Selected=0xC450F867
          DockNode  ID=0x00000010 Parent=0x00000006 SizeRef=1371,331 Selected=0xA5FE7F4E
        DockNode    ID=0x00000007 Parent=0x00000004 SizeRef=683,848 Selected=0xA3219422
      DockNode      ID=0x00000005 Parent=0x00000002 SizeRef=1370,242 Selected=0x139FDA3F
  DockNode          ID=0x0000000A Parent=0x80F5B4C5 SizeRef=550,1054 Split=Y Selected=0x3429FA32
    DockNode        ID=0x0000000B Parent=0x0000000A SizeRef=550,590 Selected=0x3429FA32
    DockNode        ID=0x0000000C Parent=0x0000000A SizeRef=550,462 Selected=0xBCE6538B
)layout";
  }

 private:
  tsd::ui::imgui::Viewport *m_viewport{nullptr};
  tsd::viskores_graph::NodeEditor *m_neditor{nullptr};
  viskores::graph::ExecutionGraph m_graph;
};

} // namespace tsd::demo

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *argv[])
{
  {
    tsd::demo::Application app(argc, argv);
    app.run(1920, 1080, "Viskores Demo App");
  }

  return 0;
}
