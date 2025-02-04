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

#include "SimulationControls.h"

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
    auto *otree = new ObjectTree(core);
    auto *solver = new SimulationControls(core);

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

    auto particles =
        ctx.createObject<tsd::Geometry>(tsd::tokens::geometry::sphere);
    particles->setName("particle_geometry");
    particles->setParameter("radius", 0.01f);

    auto blackHoles =
        ctx.createObject<tsd::Geometry>(tsd::tokens::geometry::sphere);
    blackHoles->setName("blackHole_geometry");
    blackHoles->setParameter("radius", 0.1f);

    // Colormap sampler

    auto samplerImageArray = ctx.createArray(ANARI_FLOAT32_VEC4, 3);
    auto *colorMapPtr = samplerImageArray->mapAs<tsd::float4>();
    colorMapPtr[0] = tsd::float4(0.f, 0.f, 1.f, 1.f);
    colorMapPtr[1] = tsd::float4(0.f, 1.f, 0.f, 1.f);
    colorMapPtr[2] = tsd::float4(1.f, 0.f, 0.f, 1.f);
    samplerImageArray->unmap();

    auto sampler =
        ctx.createObject<tsd::Sampler>(tsd::tokens::sampler::image1D);
    sampler->setParameter("inAttribute", "attribute0");
    sampler->setParameter("filter", "linear");
    sampler->setParameter("wrapMode", "mirrorRepeat");
    sampler->setParameterObject("image", *samplerImageArray);

    solver->setGeometry(particles, blackHoles, sampler);

    // Materials

    auto particleMat =
        ctx.createObject<tsd::Material>(tsd::tokens::material::matte);
    particleMat->setParameterObject("color", *sampler);

    auto bhMat = ctx.createObject<tsd::Material>(tsd::tokens::material::matte);
    bhMat->setParameter("color", tsd::float3(0.f));

    // Surfaces

    auto surface = ctx.createObject<tsd::Surface>();
    surface->setName("particle_surface");
    surface->setParameterObject("geometry", *particles);
    surface->setParameterObject("material", *particleMat);
    ctx.tree.insert_last_child(
        ctx.tree.root(), tsd::utility::Any(ANARI_SURFACE, surface.index()));

    surface = ctx.createObject<tsd::Surface>();
    surface->setName("bh_surface");
    surface->setParameterObject("geometry", *blackHoles);
    surface->setParameterObject("material", *bhMat);
    ctx.tree.insert_last_child(
        ctx.tree.root(), tsd::utility::Any(ANARI_SURFACE, surface.index()));

    // Setup app //

    tsd::logStatus("%s", tsd::objectDBInfo(ctx.objectDB()).c_str());
    core->tsd.sceneLoadComplete = true;

    viewport->setLibrary(core->commandLine.libraryList[0], false);

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
Pos=593,26
Size=1327,812
Collapsed=0
DockId=0x00000009,0

[Window][Object Editor]
Pos=0,742
Size=591,338
Collapsed=0
DockId=0x00000006,0

[Window][Object Tree]
Pos=0,442
Size=591,298
Collapsed=0
DockId=0x00000008,0

[Window][Log]
Pos=593,840
Size=1327,240
Collapsed=0
DockId=0x00000005,0

[Window][Simulation Controls]
Pos=0,26
Size=591,414
Collapsed=0
DockId=0x00000007,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Secondary View]
Pos=1370,26
Size=550,812
Collapsed=0
DockId=0x0000000A,0

[Table][0x39E9F5ED,1]
Column 0  Weight=1.0000

[Docking][Data]
DockSpace       ID=0x782A6D6B Window=0xDEDC5B90 Pos=0,26 Size=1920,1054 Split=X
  DockNode      ID=0x00000001 Parent=0x782A6D6B SizeRef=591,1057 Split=Y Selected=0x1FD98235
    DockNode    ID=0x00000003 Parent=0x00000001 SizeRef=549,716 Split=Y Selected=0x1FD98235
      DockNode  ID=0x00000007 Parent=0x00000003 SizeRef=549,305 Selected=0x4DDE13CE
      DockNode  ID=0x00000008 Parent=0x00000003 SizeRef=549,219 Selected=0x1FD98235
    DockNode    ID=0x00000006 Parent=0x00000001 SizeRef=549,339 Selected=0xAFC1D085
  DockNode      ID=0x00000002 Parent=0x782A6D6B SizeRef=1327,1057 Split=Y Selected=0x13926F0B
    DockNode    ID=0x00000004 Parent=0x00000002 SizeRef=1369,815 Split=X Selected=0x13926F0B
      DockNode  ID=0x00000009 Parent=0x00000004 SizeRef=775,812 CentralNode=1 Selected=0x13926F0B
      DockNode  ID=0x0000000A Parent=0x00000004 SizeRef=550,812 Selected=0xBAF13E1E
    DockNode    ID=0x00000005 Parent=0x00000002 SizeRef=1369,240 Selected=0x64F50EE5
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
    app.run(1920, 1080, "TSD Demo | Animated Particles");
  }

  return 0;
}
