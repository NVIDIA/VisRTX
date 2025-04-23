// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "BaseApplication.h"
#include "windows/DatabaseEditor.h"
#include "windows/Log.h"
#include "windows/ObjectEditor.h"
#include "windows/LayerTree.h"
#include "windows/Viewport.h"
// std
#include <chrono>

namespace tsd_viewer {

class Application : public BaseApplication
{
 public:
  Application(int argc, const char *argv[]) : BaseApplication(argc, argv) {}
  ~Application() override = default;

  anari_viewer::WindowArray setupWindows() override
  {
    auto windows = BaseApplication::setupWindows();

    auto *core = appCore();

    auto *log = new Log(core);
    auto *viewport = new Viewport(core, &m_manipulator, "Viewport");
    auto *viewport2 = new Viewport(core, &m_manipulator, "Secondary View");
    viewport2->hide();
    auto *dbeditor = new DatabaseEditor(core);
    auto *oeditor = new ObjectEditor(core);
    auto *otree = new LayerTree(core);

    windows.emplace_back(viewport);
    windows.emplace_back(viewport2);
    windows.emplace_back(dbeditor);
    windows.emplace_back(oeditor);
    windows.emplace_back(otree);
    windows.emplace_back(log);

    setWindowArray(windows);

    // Populate scene //

    m_sceneLoadFuture = std::async([vp = viewport, core = core]() {
      auto loadStart = std::chrono::steady_clock::now();
      core->setupSceneFromCommandLine();
      auto loadEnd = std::chrono::steady_clock::now();
      auto loadSeconds =
          std::chrono::duration<float>(loadEnd - loadStart).count();

      if (!core->commandLine.loadingContext) {
        tsd::logStatus("...setting up directional light");

        auto light = core->tsd.ctx.createObject<tsd::Light>(
            tsd::tokens::light::directional);
        light->setName("mainLight");
        light->setParameter("direction", tsd::float2(0.f, 240.f));

        core->tsd.ctx.defaultLayer()->root()->insert_first_child(
            tsd::utility::Any(ANARI_LIGHT, light.index()));
      }

      tsd::logStatus("...scene load complete! (%.3fs)", loadSeconds);
      tsd::logStatus("%s", tsd::objectDBInfo(core->tsd.ctx.objectDB()).c_str());
      core->tsd.sceneLoadComplete = true;

      vp->setLibrary(core->commandLine.libraryList[0], false);
    });

    return windows;
  }

  void teardown() override
  {
    if (m_sceneLoadFuture.valid())
      m_sceneLoadFuture.get();
    BaseApplication::teardown();
  }

  const char *getDefaultLayout() const override
  {
    return R"layout(
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

[Window][Layers]
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
