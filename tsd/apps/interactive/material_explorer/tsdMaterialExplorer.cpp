// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "BaseApplication.h"
#include "windows/LayerTree.h"
#include "windows/Log.h"
#include "windows/ObjectEditor.h"
#include "windows/Viewport.h"
// std
#include <vector>

namespace tsd_viewer {

class Application : public BaseApplication
{
 public:
  Application(int argc, const char *argv[]) : BaseApplication(argc, argv) {}
  ~Application() override = default;

  anari_viewer::WindowArray setupWindows() override
  {
    auto windows = BaseApplication::setupWindows();

    auto *manipulator = &m_manipulator;
    auto *core = appCore();

    auto *log = new Log(core);
    auto *viewport = new Viewport(core, manipulator, "Viewport");
    auto *viewport2 = new Viewport(core, manipulator, "Secondary View");
    if (core->commandLine.secondaryViewportLibrary.empty())
      viewport2->hide();
    auto *oeditor = new ObjectEditor(core);
    auto *otree = new LayerTree(core);

    windows.emplace_back(viewport);
    windows.emplace_back(viewport2);
    windows.emplace_back(oeditor);
    windows.emplace_back(otree);
    windows.emplace_back(log);

    setWindowArray(windows);

    // Populate scene //

    m_sceneLoadFuture = std::async(
        [viewport = viewport, vp2 = viewport2, m = manipulator, core = core]() {
          tsd::generate_material_orb(core->tsd.ctx);

          core->setupSceneFromCommandLine(true);

          if (!core->commandLine.loadingContext) {
            tsd::logStatus("...setting up directional light");

            auto light = core->tsd.ctx.createObject<tsd::Light>(
                tsd::tokens::light::directional);
            light->setName("mainLight");
            light->setParameter("direction", tsd::float2(0.f, 240.f));

            core->tsd.ctx.defaultLayer()->root()->insert_first_child(
                tsd::utility::Any(ANARI_LIGHT, light.index()));
          }

          tsd::logStatus("...scene load complete!");
          tsd::logStatus(
              "%s", tsd::objectDBInfo(core->tsd.ctx.objectDB()).c_str());
          core->tsd.sceneLoadComplete = true;

          viewport->setLibrary(core->commandLine.libraryList[0], false);
          if (!core->commandLine.secondaryViewportLibrary.empty())
            vp2->setLibrary(core->commandLine.secondaryViewportLibrary);

          m->setConfig(
              tsd::float3(0.f, 0.136f, 0.f), 0.75f, tsd::float2(330.f, 35.f));
        });

    return windows;
  }

  void teardown() override
  {
    if (m_sceneLoadFuture.valid())
      m_sceneLoadFuture.get();
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
Size=686,857
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
DockId=0x00000008,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Layers]
Pos=0,26
Size=547,302
Collapsed=0
DockId=0x00000005,0

[Window][Object Editor]
Pos=0,330
Size=547,801
Collapsed=0
DockId=0x00000006,0

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

[Table][0x413D162D,1]
Column 0  Weight=1.0000

[Docking][Data]
DockSpace       ID=0x782A6D6B Pos=0,25 Size=1920,1054 CentralNode=1 Selected=0x13926F0B
DockSpace       ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1105 Split=X
  DockNode      ID=0x00000001 Parent=0x80F5B4C5 SizeRef=547,1054 Split=Y Selected=0xCD8384B1
    DockNode    ID=0x00000005 Parent=0x00000001 SizeRef=547,288 Selected=0xCD8384B1
    DockNode    ID=0x00000006 Parent=0x00000001 SizeRef=547,764 Selected=0x82B4C496
  DockNode      ID=0x00000002 Parent=0x80F5B4C5 SizeRef=1371,1054 Split=Y Selected=0xC450F867
    DockNode    ID=0x00000007 Parent=0x00000002 SizeRef=1371,806 Split=X Selected=0xC450F867
      DockNode  ID=0x00000003 Parent=0x00000007 SizeRef=686,857 CentralNode=1 Selected=0xC450F867
      DockNode  ID=0x00000004 Parent=0x00000007 SizeRef=683,857 Selected=0xA3219422
    DockNode    ID=0x00000008 Parent=0x00000002 SizeRef=1371,246 Selected=0x139FDA3F
)layout";
  }

 private:
  std::future<void> m_sceneLoadFuture;
}; // namespace tsd_viewer

} // namespace tsd_viewer

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *argv[])
{
  {
    tsd_viewer::Application app(argc, argv);
    app.run(1920, 1080, "TSD Material Explorer");
  }

  return 0;
}
