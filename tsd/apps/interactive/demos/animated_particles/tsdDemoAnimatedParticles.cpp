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

    auto particles =
      ctx.createObject<tsd::Geometry>(tsd::tokens::geometry::sphere);
    particles->setName("particle_geometry");
    particles->setParameter("radius", 0.01f);

    auto blackHoles =
      ctx.createObject<tsd::Geometry>(tsd::tokens::geometry::sphere);
    blackHoles->setName("blackHole_geometry");
    blackHoles->setParameter("radius", 0.1f);

    solver->setGeometry(particles, blackHoles);

    auto particleMat =
      ctx.createObject<tsd::Material>(tsd::tokens::material::matte);
    particleMat->setParameter("color", tsd::float3(0.9f));

    auto bhMat =
      ctx.createObject<tsd::Material>(tsd::tokens::material::matte);
    bhMat->setParameter("color", tsd::float3(0.f));

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
Pos=0,22
Size=1200,777
Collapsed=0

[Window][Viewport]
Pos=551,22
Size=649,535
Collapsed=0
DockId=0x00000004,0

[Window][Object Editor]
Pos=0,550
Size=549,249
Collapsed=0
DockId=0x00000006,0

[Window][Object Tree]
Pos=0,329
Size=549,219
Collapsed=0
DockId=0x00000008,0

[Window][Log]
Pos=551,559
Size=649,240
Collapsed=0
DockId=0x00000005,0

[Window][Simulation Controls]
Pos=0,22
Size=549,305
Collapsed=0
DockId=0x00000007,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Table][0x39E9F5ED,1]
Column 0  Weight=1.0000

[Docking][Data]
DockSpace       ID=0x782A6D6B Window=0xDEDC5B90 Pos=0,22 Size=1200,777 Split=X
  DockNode      ID=0x00000001 Parent=0x782A6D6B SizeRef=549,1057 Split=Y Selected=0x1FD98235
    DockNode    ID=0x00000003 Parent=0x00000001 SizeRef=549,716 Split=Y Selected=0x1FD98235
      DockNode  ID=0x00000007 Parent=0x00000003 SizeRef=549,305 Selected=0x4DDE13CE
      DockNode  ID=0x00000008 Parent=0x00000003 SizeRef=549,219 Selected=0x1FD98235
    DockNode    ID=0x00000006 Parent=0x00000001 SizeRef=549,339 Selected=0xAFC1D085
  DockNode      ID=0x00000002 Parent=0x782A6D6B SizeRef=1369,1057 Split=Y Selected=0x13926F0B
    DockNode    ID=0x00000004 Parent=0x00000002 SizeRef=1369,815 CentralNode=1 Selected=0x13926F0B
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
    app.run(1200, 800, "TSD Demo | Animated Particles");
  }

  return 0;
}
