// Copyright 2024-2026 NVIDIA Corporation
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

#include "SimulationControls.h"

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
    auto *solver = new tsd::demo::SimulationControls(this);

    windows.emplace_back(viewport);
    windows.emplace_back(viewport2);
    windows.emplace_back(oeditor);
    windows.emplace_back(otree);
    windows.emplace_back(log);
    windows.emplace_back(solver);

    setWindowArray(windows);

    core->setupSceneFromCommandLine(true);

    // Populate scene data //

    // Geometry

    auto particles = scene.createObject<tsd::core::Geometry>(
        tsd::core::tokens::geometry::sphere);
    particles->setName("particle_geometry");
    particles->setParameter("radius", 0.01f);

    auto blackHoles = scene.createObject<tsd::core::Geometry>(
        tsd::core::tokens::geometry::sphere);
    blackHoles->setName("blackHole_geometry");
    blackHoles->setParameter("radius", 0.1f);

    // Colormap sampler

    auto samplerImageArray = scene.createArray(ANARI_FLOAT32_VEC4, 3);
    auto *colorMapPtr = samplerImageArray->mapAs<tsd::math::float4>();
    colorMapPtr[0] = tsd::math::float4(0.f, 0.f, 1.f, 1.f);
    colorMapPtr[1] = tsd::math::float4(0.f, 1.f, 0.f, 1.f);
    colorMapPtr[2] = tsd::math::float4(1.f, 0.f, 0.f, 1.f);
    samplerImageArray->unmap();

    auto sampler = scene.createObject<tsd::core::Sampler>(
        tsd::core::tokens::sampler::image1D);
    sampler->setParameter("inAttribute", "attribute0");
    sampler->setParameter("filter", "linear");
    sampler->setParameter("wrapMode", "mirrorRepeat");
    sampler->setParameterObject("image", *samplerImageArray);

    solver->setGeometry(particles, blackHoles, sampler);

    // Materials

    auto particleMat = scene.createObject<tsd::core::Material>(
        tsd::core::tokens::material::matte);
    particleMat->setParameterObject("color", *sampler);

    auto bhMat = scene.createObject<tsd::core::Material>(
        tsd::core::tokens::material::matte);
    bhMat->setParameter("color", tsd::math::float3(0.f));

    // Surfaces

    auto surface = scene.createObject<tsd::core::Surface>();
    surface->setName("particle_surface");
    surface->setParameterObject("geometry", *particles);
    surface->setParameterObject("material", *particleMat);
    scene.defaultLayer()->root()->insert_first_child(
        {ANARI_SURFACE, surface.index(), &core->tsd.scene});

    surface = scene.createObject<tsd::core::Surface>();
    surface->setName("bh_surface");
    surface->setParameterObject("geometry", *blackHoles);
    surface->setParameterObject("material", *bhMat);
    scene.defaultLayer()->root()->insert_first_child(
        {ANARI_SURFACE, surface.index(), &core->tsd.scene});

    // Setup app //

    tsd::core::logStatus(
        "%s", tsd::core::objectDBInfo(scene.objectDB()).c_str());
    core->tsd.sceneLoadComplete = true;

    viewport->setLibrary(core->commandLine.libraryList[0], false);

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
Pos=592,26
Size=663,863
Collapsed=0
DockId=0x00000009,0

[Window][Object Editor]
Pos=0,685
Size=590,446
Collapsed=0
DockId=0x00000006,0

[Window][Layers]
Pos=0,461
Size=590,222
Collapsed=0
DockId=0x00000005,0

[Window][Log]
Pos=592,891
Size=1328,240
Collapsed=0
DockId=0x00000008,0

[Window][Simulation Controls]
Pos=0,26
Size=590,433
Collapsed=0
DockId=0x00000003,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Secondary View]
Pos=1257,26
Size=663,863
Collapsed=0
DockId=0x0000000A,0

[Table][0x39E9F5ED,1]
Column 0  Weight=1.0000

[Table][0x413D162D,1]
Column 0  Weight=1.0000

[Docking][Data]
DockSpace       ID=0x782A6D6B Pos=0,26 Size=1920,1054 CentralNode=1
DockSpace       ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1105 Split=X
  DockNode      ID=0x00000001 Parent=0x80F5B4C5 SizeRef=590,1054 Split=Y Selected=0xCD8384B1
    DockNode    ID=0x00000003 Parent=0x00000001 SizeRef=590,413 Selected=0xDC1741DF
    DockNode    ID=0x00000004 Parent=0x00000001 SizeRef=590,639 Split=Y Selected=0xCD8384B1
      DockNode  ID=0x00000005 Parent=0x00000004 SizeRef=590,212 Selected=0xCD8384B1
      DockNode  ID=0x00000006 Parent=0x00000004 SizeRef=590,425 Selected=0x82B4C496
  DockNode      ID=0x00000002 Parent=0x80F5B4C5 SizeRef=1328,1054 Split=Y Selected=0xC450F867
    DockNode    ID=0x00000007 Parent=0x00000002 SizeRef=1328,812 Split=X Selected=0xC450F867
      DockNode  ID=0x00000009 Parent=0x00000007 SizeRef=663,863 CentralNode=1 Selected=0xC450F867
      DockNode  ID=0x0000000A Parent=0x00000007 SizeRef=663,863 Selected=0xA3219422
    DockNode    ID=0x00000008 Parent=0x00000002 SizeRef=1328,240 Selected=0x139FDA3F
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
    app.run(1920, 1080, "TSD Demo | Animated Particles");
  }

  return 0;
}
