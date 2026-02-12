// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/windows/DatabaseEditor.h>
#include <tsd/ui/imgui/windows/Log.h>
#include <tsd/ui/imgui/windows/ObjectEditor.h>
#include <tsd/ui/imgui/windows/Viewport.h>

#include "InstancingControls.h"

namespace tsd::demo {

using TSDApplication = tsd::ui::imgui::Application;
namespace tsd_ui = tsd::ui::imgui;

class Application : public TSDApplication
{
 public:
  Application() = default;
  ~Application() override = default;

  anari_viewer::WindowArray setupWindows() override
  {
    auto windows = TSDApplication::setupWindows();

    auto *core = appCore();
    auto *manipulator = &core->view.manipulator;
    core->tsd.sceneLoadComplete = true;

    auto *log = new tsd_ui::Log(this);
    auto *viewport = new tsd_ui::Viewport(this, manipulator, "Viewport");
    auto *viewport2 = new tsd_ui::Viewport(this, manipulator, "Secondary View");
    viewport2->hide();
    auto *icontrols = new tsd::demo::InstancingControls(this, "Scene Controls");
    auto *oeditor = new tsd_ui::ObjectEditor(this);
    auto *dbeditor = new tsd_ui::DatabaseEditor(this);

    windows.emplace_back(viewport);
    windows.emplace_back(viewport2);
    windows.emplace_back(icontrols);
    windows.emplace_back(oeditor);
    windows.emplace_back(log);
    windows.emplace_back(dbeditor);

    setWindowArray(windows);

    viewport->setLibraryToDefault();
    manipulator->setConfig(tsd::math::float3(2.743f, 4.747f, 0.944f),
        90.f,
        tsd::math::float2(180.f, 0.f));

    return windows;
  }

  const char *getDefaultLayout() const override
  {
    return R"layout(
[Window][MainDockSpace]
Pos=0,26
Size=1920,1105
Collapsed=0

[Window][Viewport]
Pos=549,26
Size=1371,857
Collapsed=0
DockId=0x00000003,0

[Window][Secondary View]
Pos=1237,26
Size=683,857
Collapsed=0
DockId=0x00000004,0

[Window][Log]
Pos=549,885
Size=1371,246
Collapsed=0
DockId=0x0000000A,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Layers]
Pos=0,25
Size=548,347
Collapsed=0
DockId=0x00000005,0

[Window][Object Editor]
Pos=0,609
Size=547,522
Collapsed=0
DockId=0x00000008,0

[Window][Scene Controls]
Pos=0,26
Size=547,581
Collapsed=0
DockId=0x00000007,0

[Window][Database Editor]
Pos=0,609
Size=547,522
Collapsed=0
DockId=0x00000008,1

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

[Table][0xE53C80DF,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x7FC3FA09,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xA96A74B3,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xC00D0D97,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Docking][Data]
DockSpace       ID=0x782A6D6B Pos=0,25 Size=1920,1054 Split=X Selected=0x13926F0B
  DockNode      ID=0x00000005 Parent=0x782A6D6B SizeRef=548,626 Selected=0x1FD98235
  DockNode      ID=0x00000006 Parent=0x782A6D6B SizeRef=1370,626 CentralNode=1 Selected=0x13926F0B
DockSpace       ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1105 Split=X
  DockNode      ID=0x00000001 Parent=0x80F5B4C5 SizeRef=547,1054 Split=Y Selected=0x6426B955
    DockNode    ID=0x00000007 Parent=0x00000001 SizeRef=547,581 Selected=0x6426B955
    DockNode    ID=0x00000008 Parent=0x00000001 SizeRef=547,522 Selected=0x82B4C496
  DockNode      ID=0x00000002 Parent=0x80F5B4C5 SizeRef=1371,1054 Split=Y Selected=0xC450F867
    DockNode    ID=0x00000009 Parent=0x00000002 SizeRef=1371,806 Split=X Selected=0xC450F867
      DockNode  ID=0x00000003 Parent=0x00000009 SizeRef=686,857 CentralNode=1 Selected=0xC450F867
      DockNode  ID=0x00000004 Parent=0x00000009 SizeRef=683,857 Selected=0xA3219422
    DockNode    ID=0x0000000A Parent=0x00000002 SizeRef=1371,246 Selected=0x139FDA3F
)layout";
  }
};

} // namespace tsd::demo

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *argv[])
{
  {
    tsd::demo::Application app;
    app.run(1920, 1080, "TSD Demo | Array Instancing");
  }

  return 0;
}
