// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/windows/IsosurfaceEditor.h>
#include <tsd/ui/imgui/windows/LayerTree.h>
#include <tsd/ui/imgui/windows/Log.h>
#include <tsd/ui/imgui/windows/ObjectEditor.h>
#include <tsd/ui/imgui/windows/TransferFunctionEditor.h>
#include <tsd/ui/imgui/windows/Viewport.h>
// std
#include <algorithm>
#include <vector>

#include "SolverControls.h"

namespace tsd::demo {

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
    auto &scene = core->tsd.scene;
    auto *manipulator = &core->view.manipulator;

    auto *log = new tsd_ui::Log(this);
    auto *viewport = new tsd_ui::Viewport(this, manipulator, "Viewport");
    auto *viewport2 = new tsd_ui::Viewport(this, manipulator, "Secondary View");
    viewport2->hide();
    auto *oeditor = new tsd_ui::ObjectEditor(this);
    auto *otree = new tsd_ui::LayerTree(this);
    auto *tfeditor = new tsd_ui::TransferFunctionEditor(this);
    auto *isoeditor = new tsd_ui::IsosurfaceEditor(this);
    auto *solver = new tsd::demo::SolverControls(this);

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

    auto colorArray = core->tsd.scene.createArray(ANARI_FLOAT32_VEC4, 256);
    colorArray->setData(
        tsd::core::makeDefaultColorMap(colorArray->size()).data());

    auto field = scene.createObject<tsd::core::SpatialField>(
        tsd::core::tokens::spatial_field::structuredRegular);
    field->setName("jacobi_field");
    solver->setField(field);

    auto volume = scene.createObject<tsd::core::Volume>(
        tsd::core::tokens::volume::transferFunction1D);
    volume->setName("jacobi_volume");

    tsd::math::float2 valueRange{0.f, 1.f};
    if (field)
      valueRange = field->computeValueRange();
    volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
    volume->setParameterObject("value", *field);
    volume->setParameterObject("color", *colorArray);

    auto volumeNode = scene.defaultLayer()->root()->insert_first_child(
        {ANARI_VOLUME, volume.index(), &core->tsd.scene});

    // Setup app //

    core->setSelected(volumeNode);

    tsd::core::logStatus(
        "%s", tsd::core::objectDBInfo(scene.objectDB()).c_str());
    core->tsd.sceneLoadComplete = true;

    viewport->setLibrary(core->commandLine.libraryList[0], false);

    solver->setUpdateCallback([=]() mutable {
      auto valueRange = field->computeValueRange();
      volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
    });

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
Size=821,429
Collapsed=0
DockId=0x00000003,0

[Window][Secondary View]
Pos=549,457
Size=821,426
Collapsed=0
DockId=0x00000004,0

[Window][Log]
Pos=549,885
Size=821,246
Collapsed=0
DockId=0x0000000E,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Layers]
Pos=0,381
Size=547,205
Collapsed=0
DockId=0x0000000C,0

[Window][Object Editor]
Pos=0,588
Size=547,543
Collapsed=0
DockId=0x0000000D,0

[Window][TF Editor]
Pos=1372,26
Size=548,553
Collapsed=0
DockId=0x00000005,0

[Window][Isosurface Editor]
Pos=1372,581
Size=548,550
Collapsed=0
DockId=0x00000006,0

[Window][Solver Controls]
Pos=0,26
Size=547,353
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

[Table][0x9D05AA94,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xA7A1FF83,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Docking][Data]
DockSpace         ID=0x782A6D6B Pos=0,22 Size=1920,1057 CentralNode=1 Selected=0x13926F0B
DockSpace         ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1105 Split=X
  DockNode        ID=0x0000000A Parent=0x80F5B4C5 SizeRef=547,1054 Split=Y Selected=0xF64741D0
    DockNode      ID=0x00000008 Parent=0x0000000A SizeRef=547,337 Selected=0xF64741D0
    DockNode      ID=0x00000009 Parent=0x0000000A SizeRef=547,715 Split=Y Selected=0xCD8384B1
      DockNode    ID=0x0000000C Parent=0x00000009 SizeRef=547,195 Selected=0xCD8384B1
      DockNode    ID=0x0000000D Parent=0x00000009 SizeRef=547,518 Selected=0x82B4C496
  DockNode        ID=0x0000000B Parent=0x80F5B4C5 SizeRef=1371,1054 Split=X
    DockNode      ID=0x00000001 Parent=0x0000000B SizeRef=1370,1054 Split=Y Selected=0xC450F867
      DockNode    ID=0x00000007 Parent=0x00000001 SizeRef=821,806 Split=Y Selected=0xC450F867
        DockNode  ID=0x00000003 Parent=0x00000007 SizeRef=821,429 CentralNode=1 Selected=0xC450F867
        DockNode  ID=0x00000004 Parent=0x00000007 SizeRef=821,426 Selected=0xA3219422
      DockNode    ID=0x0000000E Parent=0x00000001 SizeRef=821,246 Selected=0x139FDA3F
    DockNode      ID=0x00000002 Parent=0x0000000B SizeRef=548,1054 Split=Y Selected=0x3429FA32
      DockNode    ID=0x00000005 Parent=0x00000002 SizeRef=548,527 Selected=0x3429FA32
      DockNode    ID=0x00000006 Parent=0x00000002 SizeRef=548,525 Selected=0xBCE6538B
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
    tsd::demo::Application app(argc, argv);
    app.run(1920, 1080, "TSD Demo | Animated Volume");
  }

  return 0;
}
