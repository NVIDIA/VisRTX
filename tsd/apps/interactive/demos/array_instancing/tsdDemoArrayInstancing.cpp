// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "BaseApplication.h"
#include "windows/Log.h"
#include "windows/ObjectEditor.h"
#include "windows/Viewport.h"

#include "InstancingControls.h"

namespace tsd_viewer {

class Application : public BaseApplication
{
 public:
  Application() = default;
  ~Application() override = default;

  anari_viewer::WindowArray setupWindows() override
  {
    auto windows = BaseApplication::setupWindows();

    auto *core = appCore();
    core->tsd.sceneLoadComplete = true;

    auto *log = new Log(core);
    auto *viewport = new Viewport(core, &m_manipulator, "Viewport");
    auto *viewport2 = new Viewport(core, &m_manipulator, "Secondary View");
    viewport2->hide();
    auto *icontrols = new InstancingControls(core, "Scene Controls");
    auto *oeditor = new ObjectEditor(core);

    windows.emplace_back(viewport);
    windows.emplace_back(viewport2);
    windows.emplace_back(icontrols);
    windows.emplace_back(oeditor);
    windows.emplace_back(log);

    setWindowArray(windows);

    viewport->setLibrary(core->commandLine.libraryList[0], false);
    m_manipulator.setConfig(
        tsd::float3(2.743f, 4.747f, 0.944f), 90.f, tsd::float2(180.f, 0.f));

    return windows;
  }

  const char *getDefaultLayout() const override
  {
    return R"layout(
[Window][MainDockSpace]
Pos=0,25
Size=1920,1054
Collapsed=0

[Window][Viewport]
Pos=550,25
Size=1370,806
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

[Window][Layers]
Pos=0,25
Size=548,347
Collapsed=0
DockId=0x00000007,0

[Window][Object Editor]
Pos=0,551
Size=548,528
Collapsed=0
DockId=0x0000000A,0

[Window][Scene Controls]
Pos=0,25
Size=548,524
Collapsed=0
DockId=0x00000009,0

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

[Docking][Data]
DockSpace       ID=0x782A6D6B Window=0xDEDC5B90 Pos=0,25 Size=1920,1054 Split=X Selected=0x13926F0B
  DockNode      ID=0x00000005 Parent=0x782A6D6B SizeRef=548,626 Split=Y Selected=0x1FD98235
    DockNode    ID=0x00000007 Parent=0x00000005 SizeRef=549,288 Selected=0x1FD98235
    DockNode    ID=0x00000008 Parent=0x00000005 SizeRef=549,584 Split=Y Selected=0xAFC1D085
      DockNode  ID=0x00000009 Parent=0x00000008 SizeRef=548,524 Selected=0xD82CEBDE
      DockNode  ID=0x0000000A Parent=0x00000008 SizeRef=548,528 Selected=0xAFC1D085
  DockNode      ID=0x00000006 Parent=0x782A6D6B SizeRef=1370,626 Split=Y Selected=0x13926F0B
    DockNode    ID=0x00000001 Parent=0x00000006 SizeRef=1049,626 Split=X Selected=0x13926F0B
      DockNode  ID=0x00000003 Parent=0x00000001 SizeRef=684,626 CentralNode=1 Selected=0x13926F0B
      DockNode  ID=0x00000004 Parent=0x00000001 SizeRef=684,626 Selected=0xBAF13E1E
    DockNode    ID=0x00000002 Parent=0x00000006 SizeRef=1049,246 Selected=0x64F50EE5
)layout";
  }
};

} // namespace tsd_viewer

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *argv[])
{
  {
    tsd_viewer::Application app;
    app.run(1920, 1080, "TSD Demo | Array Instancing");
  }

  return 0;
}
