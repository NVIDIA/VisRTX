// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "BaseApplication.h"
#include "windows/DatabaseEditor.h"
#include "windows/LayerTree.h"
#include "windows/Log.h"
#include "windows/ObjectEditor.h"
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
    if (core->commandLine.secondaryViewportLibrary.empty())
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

    auto populateScene = [vp = viewport, vp2 = viewport2, core = core]() {
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
      if (!core->commandLine.secondaryViewportLibrary.empty())
        vp2->setLibrary(core->commandLine.secondaryViewportLibrary);
    };

#if 1
    m_sceneLoadFuture = std::async(populateScene);
#else
    populateScene();
#endif

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
Pos=0,26
Size=1920,1105
Collapsed=0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Viewport]
Pos=550,26
Size=685,848
Collapsed=0
DockId=0x00000006,0

[Window][Database Editor]
Pos=0,552
Size=549,528
Collapsed=0
DockId=0x00000003,1

[Window][Layers]
Pos=0,26
Size=548,1105
Collapsed=0
DockId=0x00000001,0

[Window][Object Editor]
Pos=0,552
Size=549,528
Collapsed=0
DockId=0x00000003,0

[Window][Log]
Pos=550,876
Size=1370,255
Collapsed=0
DockId=0x00000005,0

[Window][Secondary View]
Pos=1237,26
Size=683,848
Collapsed=0
DockId=0x00000007,0

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
DockNode        ID=0x00000003 Pos=0,552 Size=549,528 Selected=0x82B4C496
DockSpace       ID=0x782A6D6B Pos=0,26 Size=1920,1054 CentralNode=1
DockSpace       ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1105 Split=X
  DockNode      ID=0x00000001 Parent=0x80F5B4C5 SizeRef=548,1054 Selected=0xCD8384B1
  DockNode      ID=0x00000002 Parent=0x80F5B4C5 SizeRef=1370,1054 Split=Y Selected=0xC450F867
    DockNode    ID=0x00000004 Parent=0x00000002 SizeRef=1370,797 Split=X Selected=0xC450F867
      DockNode  ID=0x00000006 Parent=0x00000004 SizeRef=685,848 CentralNode=1 Selected=0xC450F867
      DockNode  ID=0x00000007 Parent=0x00000004 SizeRef=683,848 Selected=0xA3219422
    DockNode    ID=0x00000005 Parent=0x00000002 SizeRef=1370,255 Selected=0x139FDA3F
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
