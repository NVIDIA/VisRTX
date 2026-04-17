// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/windows/Animations.h>
#include <tsd/ui/imgui/windows/DatabaseEditor.h>
#include <tsd/ui/imgui/windows/IsosurfaceEditor.h>
#include <tsd/ui/imgui/windows/LayerTree.h>
#include <tsd/ui/imgui/windows/Log.h>
#include <tsd/ui/imgui/windows/ObjectEditor.h>
#include <tsd/ui/imgui/windows/Timeline.h>
#include <tsd/ui/imgui/windows/TransferFunctionEditor.h>
#include <tsd/ui/imgui/windows/Viewport.h>

#include "WeightedPointsControls.h"

#include <string>

namespace tsd::demo {

using TSDApplication = tsd::ui::imgui::Application;
namespace tsd_ui = tsd::ui::imgui;

class Application : public TSDApplication
{
 public:
  Application(const std::string &pdbPath)
      : m_pdbPath(pdbPath)
  {}

  ~Application() override = default;

  anari_viewer::WindowArray setupWindows() override
  {
    auto windows = TSDApplication::setupWindows();

    auto *ctx = appContext();
    auto *manipulator = &ctx->view.manipulator;
    ctx->tsd.sceneLoadComplete = true;

    auto *viewport = new tsd_ui::Viewport(this, manipulator, "Viewport");
    auto *wpcontrols =
        new WeightedPointsControls(this, "Custom Field", m_pdbPath);
    auto *otree = new tsd_ui::LayerTree(this);
    auto *tfeditor = new tsd_ui::TransferFunctionEditor(this);
    auto *isoeditor = new tsd_ui::IsosurfaceEditor(this);
    auto *oeditor = new tsd_ui::ObjectEditor(this);
    auto *dbeditor = new tsd_ui::DatabaseEditor(this);
    auto *timeline = new tsd_ui::Timeline(this);
    auto *animations = new tsd_ui::Animations(this);
    auto *log = new tsd_ui::Log(this);

    windows.emplace_back(viewport);
    windows.emplace_back(wpcontrols);
    windows.emplace_back(otree);
    windows.emplace_back(tfeditor);
    windows.emplace_back(isoeditor);
    windows.emplace_back(oeditor);
    windows.emplace_back(dbeditor);
    windows.emplace_back(timeline);
    windows.emplace_back(animations);
    windows.emplace_back(log);

    setWindowArray(windows);

    isoeditor->hide();

    viewport->setLibraryToDefault();
    manipulator->setConfig(tsd::math::float3(0.f, 0.f, 0.f),
        3.5f,
        tsd::math::float2(30.f, -20.f));

    return windows;
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
Pos=449,26
Size=1021,611
Collapsed=0
DockId=0x00000003,0

[Window][Timeline]
Pos=449,639
Size=1021,220
Collapsed=0
DockId=0x00000010,0

[Window][Log]
Pos=449,861
Size=1471,270
Collapsed=0
DockId=0x0000000A,0

[Window][Layers]
Pos=0,26
Size=447,581
Collapsed=0
DockId=0x00000007,0

[Window][Custom Field]
Pos=0,26
Size=447,581
Collapsed=0
DockId=0x00000007,1

[Window][Object Editor]
Pos=0,609
Size=447,522
Collapsed=0
DockId=0x00000008,0

[Window][Database Editor]
Pos=0,609
Size=447,522
Collapsed=0
DockId=0x00000008,1

[Window][Isosurface Editor]
Pos=0,609
Size=447,522
Collapsed=0
DockId=0x00000008,2

[Window][Animations]
Pos=0,609
Size=447,522
Collapsed=0
DockId=0x00000008,3

[Window][TF Editor]
Pos=1472,26
Size=448,611
Collapsed=0
DockId=0x0000000C,0

[Docking][Data]
DockSpace       ID=0x782A6D6B Pos=0,25 Size=1920,1054 Split=X Selected=0x13926F0B
  DockNode      ID=0x00000005 Parent=0x782A6D6B SizeRef=448,626 Selected=0x1FD98235
  DockNode      ID=0x00000006 Parent=0x782A6D6B SizeRef=1470,626 CentralNode=1 Selected=0x13926F0B
DockSpace       ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1105 Split=X
  DockNode      ID=0x00000001 Parent=0x80F5B4C5 SizeRef=447,1054 Split=Y Selected=0xCD8384B1
    DockNode    ID=0x00000007 Parent=0x00000001 SizeRef=447,581 Selected=0xCD8384B1
    DockNode    ID=0x00000008 Parent=0x00000001 SizeRef=447,522 Selected=0x82B4C496
  DockNode      ID=0x00000002 Parent=0x80F5B4C5 SizeRef=1471,1054 Split=Y Selected=0xC450F867
    DockNode    ID=0x0000000B Parent=0x00000002 SizeRef=1471,833 Split=X Selected=0xC450F867
      DockNode  ID=0x0000000F Parent=0x0000000B SizeRef=1021,833 Split=Y Selected=0xC450F867
        DockNode ID=0x00000003 Parent=0x0000000F SizeRef=1021,611 CentralNode=1 Selected=0xC450F867
        DockNode ID=0x00000010 Parent=0x0000000F SizeRef=1021,220 Selected=0xB86F32B9
      DockNode  ID=0x0000000C Parent=0x0000000B SizeRef=448,833 Selected=0x3429FA32
    DockNode    ID=0x0000000A Parent=0x00000002 SizeRef=1471,270 Selected=0x139FDA3F
)layout";
  }

 private:
  std::string m_pdbPath;
};

} // namespace tsd::demo

///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *argv[])
{
  std::string pdbPath;
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    if ((arg == "--pdb" || arg == "-pdb") && i + 1 < argc)
      pdbPath = argv[++i];
    else if (pdbPath.empty() && !arg.empty() && arg[0] != '-')
      pdbPath = arg;
  }

  {
    tsd::demo::Application app(pdbPath);
    app.run(1920, 1080, "TSD Demo | Custom Field");
  }

  return 0;
}
