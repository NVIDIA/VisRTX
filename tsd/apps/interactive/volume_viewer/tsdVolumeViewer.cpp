// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "BaseApplication.h"
#include "windows/IsosurfaceEditor.h"
#include "windows/Log.h"
#include "windows/ObjectEditor.h"
#include "windows/LayerTree.h"
#include "windows/TransferFunctionEditor.h"
#include "windows/Viewport.h"
// std
#include <algorithm>
#include <vector>

static std::string g_filename;

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
    viewport2->hide();
    auto *oeditor = new ObjectEditor(core);
    auto *otree = new LayerTree(core);
    auto *tfeditor = new TransferFunctionEditor(core);
    auto *isoeditor = new IsosurfaceEditor(core);

    windows.emplace_back(viewport);
    windows.emplace_back(viewport2);
    windows.emplace_back(oeditor);
    windows.emplace_back(otree);
    windows.emplace_back(log);
    windows.emplace_back(tfeditor);
    windows.emplace_back(isoeditor);

    setWindowArray(windows);

    // Populate scene //

    m_sceneLoadFuture = std::async([viewport = viewport,
                                       m = manipulator,
                                       tf = tfeditor,
                                       core = core]() {
      auto colorMapArray = core->tsd.ctx.createArray(ANARI_FLOAT32_VEC4, 256);

      tsd::VolumeRef volume;

      if (!g_filename.empty()) {
        volume = tsd::import_volume(
            core->tsd.ctx, g_filename.c_str(), colorMapArray);
      }

      if (!volume) {
        if (!g_filename.empty())
          tsd::logWarning("unable to load volume from file, using placeholder");

        auto tx1 = core->tsd.ctx.insertChildTransformNode(
            core->tsd.ctx.defaultLayer()->root());

        volume = tsd::generate_noiseVolume(core->tsd.ctx, tx1, colorMapArray);
      }

      core->tsd.selectedObject = volume.data();
      tf->setValueRange(volume->parameter("valueRange")
              ->value()
              .getAs<tsd::float2>(ANARI_FLOAT32_BOX1));

      core->setupSceneFromCommandLine(true);

      if (!core->commandLine.loadingContext) {
        tsd::logStatus("...setting up directional light");

        auto light = core->tsd.ctx.createObject<tsd::Light>(
            tsd::tokens::light::directional);
        light->setName("mainLight");
        light->setParameter("direction", tsd::float2(0.f, 240.f));

        core->tsd.ctx.defaultLayer()->insert_first_child(
            core->tsd.ctx.defaultLayer()->root(),
            tsd::utility::Any(ANARI_LIGHT, light.index()));
      }

      tsd::logStatus("...scene load complete!");
      tsd::logStatus("%s", tsd::objectDBInfo(core->tsd.ctx.objectDB()).c_str());
      core->tsd.sceneLoadComplete = true;

      viewport->setLibrary(core->commandLine.libraryList[0], false);

      tf->setUpdateCallback([=](const tsd::float2 &valueRange,
                                const std::vector<tsd::float4> &co) mutable {
        auto *colorMap = colorMapArray->mapAs<tsd::float4>();
        std::copy(co.begin(), co.end(), colorMap);
        colorMapArray->unmap();
        volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
      });
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
Pos=0,26
Size=1920,1054
Collapsed=0

[Window][Viewport]
Pos=507,26
Size=915,762
Collapsed=0
DockId=0x0000000B,0

[Window][Secondary View]
Pos=1236,25
Size=684,806
Collapsed=0
DockId=0x00000004,0

[Window][Log]
Pos=507,790
Size=915,290
Collapsed=0
DockId=0x0000000C,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Layers]
Pos=0,26
Size=505,233
Collapsed=0
DockId=0x00000005,0

[Window][Object Editor]
Pos=0,261
Size=505,819
Collapsed=0
DockId=0x00000006,0

[Window][TF Editor]
Pos=1424,26
Size=496,502
Collapsed=0
DockId=0x00000009,0

[Window][Isosurface Editor]
Pos=1424,530
Size=496,550
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

[Table][0x75C219FA,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xE1897A69,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x413D162D,1]
Column 0  Weight=1.0000

[Table][0x3ABA3223,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x48750CD3,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Docking][Data]
DockSpace       ID=0x782A6D6B Pos=0,22 Size=2489,1211 Split=X Selected=0x13926F0B
  DockNode      ID=0x00000003 Parent=0x782A6D6B SizeRef=684,626 CentralNode=1 Selected=0x13926F0B
  DockNode      ID=0x00000004 Parent=0x782A6D6B SizeRef=684,626 Selected=0xBAF13E1E
DockSpace       ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1054 Split=X
  DockNode      ID=0x00000007 Parent=0x80F5B4C5 SizeRef=1422,1413 Split=X
    DockNode    ID=0x00000001 Parent=0x00000007 SizeRef=505,1413 Split=Y Selected=0xCD8384B1
      DockNode  ID=0x00000005 Parent=0x00000001 SizeRef=547,313 Selected=0xCD8384B1
      DockNode  ID=0x00000006 Parent=0x00000001 SizeRef=547,1098 Selected=0x82B4C496
    DockNode    ID=0x00000002 Parent=0x00000007 SizeRef=915,1413 Split=Y Selected=0xC450F867
      DockNode  ID=0x0000000B Parent=0x00000002 SizeRef=1595,762 CentralNode=1 Selected=0xC450F867
      DockNode  ID=0x0000000C Parent=0x00000002 SizeRef=1595,290 Selected=0x139FDA3F
  DockNode      ID=0x00000008 Parent=0x80F5B4C5 SizeRef=496,1413 Split=Y Selected=0x3429FA32
    DockNode    ID=0x00000009 Parent=0x00000008 SizeRef=548,673 Selected=0x3429FA32
    DockNode    ID=0x0000000A Parent=0x00000008 SizeRef=548,738 Selected=0xBCE6538B
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
    if (argc > 1)
      g_filename = argv[1];

    tsd_viewer::Application app(argc, argv);
    app.run(1920, 1080, "TSD Volume Viewer");
  }

  return 0;
}
