// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_io
#include <tsd/io/procedural.hpp>
// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/windows/Animations.h>
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

namespace tsd_viewer {

using TSDApplication = tsd::ui::imgui::Application;
namespace tsd_ui = tsd::ui::imgui;

class Application : public TSDApplication
{
 public:
  Application(int argc, const char *argv[]) : TSDApplication(argc, argv) {}
  ~Application() override = default;

  anari_viewer::WindowArray setupWindows() override
  {
    auto windows = TSDApplication::setupWindows();

    auto *core = appCore();

    auto *viewport =
        new tsd_ui::Viewport(this, &core->view.manipulator, "Viewport");
    auto *viewport2 =
        new tsd_ui::Viewport(this, &core->view.manipulator, "Secondary View");
    auto *animations = new tsd_ui::Animations(this);
    auto *cameras = new tsd_ui::CameraPoses(this, viewport);
    auto *log = new tsd_ui::Log(this);
    auto *dbeditor = new tsd_ui::DatabaseEditor(this);
    auto *oeditor = new tsd_ui::ObjectEditor(this);
    auto *otree = new tsd_ui::LayerTree(this);
    auto *tfeditor = new tsd_ui::TransferFunctionEditor(this);
    auto *isoeditor = new tsd_ui::IsosurfaceEditor(this);

    windows.emplace_back(animations);
    windows.emplace_back(cameras);
    windows.emplace_back(viewport);
    windows.emplace_back(viewport2);
    windows.emplace_back(dbeditor);
    windows.emplace_back(oeditor);
    windows.emplace_back(otree);
    windows.emplace_back(log);
    windows.emplace_back(tfeditor);
    windows.emplace_back(isoeditor);

    setWindowArray(windows);

    if (core->commandLine.secondaryViewportLibrary.empty())
      viewport2->hide();
    tfeditor->hide();
    isoeditor->hide();

    // Populate scene //

    auto populateScene = [vp = viewport, vp2 = viewport2, core = core]() {
      auto loadStart = std::chrono::steady_clock::now();
      core->setupSceneFromCommandLine();
      auto loadEnd = std::chrono::steady_clock::now();
      auto loadSeconds =
          std::chrono::duration<float>(loadEnd - loadStart).count();

      auto &scene = core->tsd.scene;

      const bool setupDefaultLight = !core->commandLine.loadedFromStateFile
          && scene.numberOfObjects(ANARI_LIGHT) == 0;
      if (setupDefaultLight) {
        tsd::core::logStatus("...setting up default light");
        tsd::io::generate_default_lights(scene);
      }

      tsd::core::logStatus("...scene load complete! (%.3fs)", loadSeconds);
      tsd::core::logStatus(
          "%s", tsd::core::objectDBInfo(scene.objectDB()).c_str());
      core->tsd.sceneLoadComplete = true;

      if (!core->commandLine.loadedFromStateFile) {
        vp->setLibrary(core->commandLine.libraryList[0], false);
        if (!core->commandLine.secondaryViewportLibrary.empty())
          vp2->setLibrary(core->commandLine.secondaryViewportLibrary);
      }
    };

#if 1
    m_taskModal->activate(populateScene, "Please Wait: Loading Scene...");
#else
    populateScene();
#endif

    return windows;
  }

  void teardown() override
  {
    if (m_sceneLoadFuture.valid())
      m_sceneLoadFuture.get();
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
Pos=549,26
Size=1371,848
Collapsed=0
DockId=0x00000006,0

[Window][Database Editor]
Pos=0,603
Size=547,528
Collapsed=0
DockId=0x00000009,1

[Window][Layers]
Pos=0,26
Size=547,575
Collapsed=0
DockId=0x00000008,0

[Window][Object Editor]
Pos=0,603
Size=547,528
Collapsed=0
DockId=0x00000009,0

[Window][Log]
Pos=549,876
Size=1371,255
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
Size=547,575
Collapsed=0
DockId=0x00000008,2

[Window][Animations]
Pos=0,26
Size=547,575
Collapsed=0
DockId=0x00000008,1

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
DockSpace         ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1105 Split=X
  DockNode        ID=0x00000003 Parent=0x80F5B4C5 SizeRef=1368,1054 Split=X
    DockNode      ID=0x00000001 Parent=0x00000003 SizeRef=547,1105 Split=Y Selected=0xCD8384B1
      DockNode    ID=0x00000008 Parent=0x00000001 SizeRef=547,575 Selected=0xCD8384B1
      DockNode    ID=0x00000009 Parent=0x00000001 SizeRef=547,528 Selected=0x82B4C496
    DockNode      ID=0x00000002 Parent=0x00000003 SizeRef=1371,1105 Split=Y
      DockNode    ID=0x00000004 Parent=0x00000002 SizeRef=1370,797 Split=X Selected=0xC450F867
        DockNode  ID=0x00000006 Parent=0x00000004 SizeRef=685,848 CentralNode=1 Selected=0xC450F867
        DockNode  ID=0x00000007 Parent=0x00000004 SizeRef=683,848 Selected=0xA3219422
      DockNode    ID=0x00000005 Parent=0x00000002 SizeRef=1370,255 Selected=0x139FDA3F
  DockNode        ID=0x0000000A Parent=0x80F5B4C5 SizeRef=550,1054 Split=Y Selected=0x3429FA32
    DockNode      ID=0x0000000B Parent=0x0000000A SizeRef=550,590 Selected=0x3429FA32
    DockNode      ID=0x0000000C Parent=0x0000000A SizeRef=550,462 Selected=0xBCE6538B
)layout";
  }

 private:
  std::future<void> m_sceneLoadFuture;
};

} // namespace tsd_viewer

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *argv[])
{
  {
    tsd_viewer::Application app(argc, argv);
    app.run(1920, 1080, "TSD Viewer");
  }

  return 0;
}
