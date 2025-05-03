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

#include "SolverControls.h"

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
    auto &ctx = core->tsd.ctx;

    auto *log = new Log(core);
    auto *viewport = new Viewport(core, manipulator, "Viewport");
    auto *viewport2 = new Viewport(core, manipulator, "Secondary View");
    viewport2->hide();
    auto *oeditor = new ObjectEditor(core);
    auto *otree = new LayerTree(core);
    auto *tfeditor = new TransferFunctionEditor(core);
    auto *isoeditor = new IsosurfaceEditor(core);
    auto *solver = new SolverControls(core);

    windows.emplace_back(viewport);
    windows.emplace_back(viewport2);
    windows.emplace_back(oeditor);
    windows.emplace_back(otree);
    windows.emplace_back(log);
    windows.emplace_back(tfeditor);
    windows.emplace_back(isoeditor);
    windows.emplace_back(solver);

    setWindowArray(windows);

    core->setupSceneFromCommandLine(true);

    // Populate scene data //

    auto colorArray = ctx.createArray(ANARI_FLOAT32_VEC3, 256);
    auto opacityArray = ctx.createArray(ANARI_FLOAT32, 256);

    auto field = ctx.createObject<tsd::SpatialField>(
        tsd::tokens::spatial_field::structuredRegular);
    field->setName("jacobi_field");
    solver->setField(field);

    auto volume =
        ctx.createObject<tsd::Volume>(tsd::tokens::volume::transferFunction1D);
    volume->setName("jacobi_volume");

    tsd::float2 valueRange{0.f, 1.f};
    if (field)
      valueRange = field->computeValueRange();
    volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
    volume->setParameterObject("value", *field);
    volume->setParameterObject("color", *colorArray);
    volume->setParameterObject("opacity", *opacityArray);

    ctx.defaultLayer()->root()->insert_first_child(
        tsd::utility::Any(ANARI_VOLUME, volume.index()));

    // Setup app //

    core->tsd.selectedObject = volume.data();
    tfeditor->setValueRange(volume->parameter("valueRange")
            ->value()
            .getAs<tsd::float2>(ANARI_FLOAT32_BOX1));

    tsd::logStatus("%s", tsd::objectDBInfo(ctx.objectDB()).c_str());
    core->tsd.sceneLoadComplete = true;

    viewport->setLibrary(core->commandLine.libraryList[0], false);

    tfeditor->setUpdateCallback(
        [=](const tsd::float2 &valueRange,
            const std::vector<tsd::float4> &co) mutable {
          auto *colors = colorArray->mapAs<tsd::float3>();
          auto *opacities = opacityArray->mapAs<float>();
          std::transform(
              co.begin(), co.end(), colors, [](const tsd::float4 &v) {
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

    solver->setUpdateCallback(
        [=]() mutable { tfeditor->setValueRange(field->computeValueRange()); });

    return windows;
  }

  const char *getDefaultLayout() const override
  {
    return R"layout(
[Window][MainDockSpace]
Pos=0,26
Size=1920,1054
Collapsed=0

[Window][Viewport]
Pos=549,26
Size=821,806
Collapsed=0
DockId=0x00000007,0

[Window][Secondary View]
Pos=1236,25
Size=684,806
Collapsed=0
DockId=0x00000004,0

[Window][Log]
Pos=549,834
Size=821,246
Collapsed=0
DockId=0x0000000E,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Layers]
Pos=0,365
Size=547,195
Collapsed=0
DockId=0x0000000C,0

[Window][Object Editor]
Pos=0,562
Size=547,518
Collapsed=0
DockId=0x0000000D,0

[Window][TF Editor]
Pos=1372,26
Size=548,527
Collapsed=0
DockId=0x00000005,0

[Window][Isosurface Editor]
Pos=1372,555
Size=548,525
Collapsed=0
DockId=0x00000006,0

[Window][Solver Controls]
Pos=0,26
Size=547,337
Collapsed=0
DockId=0x00000008,0

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

[Table][0x9F2DA3B7,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x930141F5,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x9D58ADAC,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xAE630623,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x413D162D,1]
Column 0  Weight=1.0000

[Docking][Data]
DockSpace       ID=0x782A6D6B Pos=0,22 Size=1920,1057 Split=X Selected=0x13926F0B
  DockNode      ID=0x00000003 Parent=0x782A6D6B SizeRef=684,626 CentralNode=1 Selected=0x13926F0B
  DockNode      ID=0x00000004 Parent=0x782A6D6B SizeRef=684,626 Selected=0xBAF13E1E
DockSpace       ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1054 Split=X
  DockNode      ID=0x0000000A Parent=0x80F5B4C5 SizeRef=547,1054 Split=Y Selected=0xF64741D0
    DockNode    ID=0x00000008 Parent=0x0000000A SizeRef=547,337 Selected=0xF64741D0
    DockNode    ID=0x00000009 Parent=0x0000000A SizeRef=547,715 Split=Y Selected=0xCD8384B1
      DockNode  ID=0x0000000C Parent=0x00000009 SizeRef=547,195 Selected=0xCD8384B1
      DockNode  ID=0x0000000D Parent=0x00000009 SizeRef=547,518 Selected=0x82B4C496
  DockNode      ID=0x0000000B Parent=0x80F5B4C5 SizeRef=1371,1054 Split=X
    DockNode    ID=0x00000001 Parent=0x0000000B SizeRef=1370,1054 Split=Y Selected=0xC450F867
      DockNode  ID=0x00000007 Parent=0x00000001 SizeRef=821,806 CentralNode=1 Selected=0xC450F867
      DockNode  ID=0x0000000E Parent=0x00000001 SizeRef=821,246 Selected=0x139FDA3F
    DockNode    ID=0x00000002 Parent=0x0000000B SizeRef=548,1054 Split=Y Selected=0x3429FA32
      DockNode  ID=0x00000005 Parent=0x00000002 SizeRef=548,527 Selected=0x3429FA32
      DockNode  ID=0x00000006 Parent=0x00000002 SizeRef=548,525 Selected=0xBCE6538B
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
    tsd_viewer::Application app(argc, argv);
    app.run(1920, 1080, "TSD Demo | Animated Volume");
  }

  return 0;
}
