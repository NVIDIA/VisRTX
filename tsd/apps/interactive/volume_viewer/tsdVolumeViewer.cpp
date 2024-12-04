// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "BaseApplication.h"
#include "windows/IsosurfaceEditor.h"
#include "windows/Log.h"
#include "windows/ObjectEditor.h"
#include "windows/ObjectTree.h"
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
    auto *otree = new ObjectTree(core);
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
      auto colorArray = core->tsd.ctx.createArray(ANARI_FLOAT32_VEC3, 256);
      auto opacityArray = core->tsd.ctx.createArray(ANARI_FLOAT32, 256);

      tsd::VolumeRef volume;

      if (!g_filename.empty()) {
        volume = tsd::import_volume(
            core->tsd.ctx, g_filename.c_str(), colorArray, opacityArray);
      }

      if (!volume) {
        if (!g_filename.empty())
          tsd::logWarning("unable to load volume from file, using placeholder");

        volume = tsd::generate_noiseVolume(
            core->tsd.ctx, core->tsd.ctx.tree.root(), colorArray, opacityArray);
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

        core->tsd.ctx.tree.insert_first_child(core->tsd.ctx.tree.root(),
            tsd::utility::Any(ANARI_LIGHT, light.index()));
      }

      tsd::logStatus("...scene load complete!");
      tsd::logStatus("%s", tsd::objectDBInfo(core->tsd.ctx.objectDB()).c_str());
      core->tsd.sceneLoadComplete = true;

      viewport->setLibrary(core->commandLine.libraryList[0], false);

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
