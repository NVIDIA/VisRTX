// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_app
#include <tsd/app/Core.h>
// tsd_ui_imgui
#include <tsd/ui/imgui/windows/Log.h>
// tsd_io
#include <tsd/io/importers.hpp>
#include <tsd/io/procedural.hpp>
// std
#include <iostream>
#include <random>
#include <vector>

#include "DistributedViewport.h"
#include "ViewState.h"

using float2 = tsd::math::float2;
using float3 = tsd::math::float3;
using float4 = tsd::math::float4;

using tsd::app::ImporterType;

int g_rank = -1;

static bool g_useDefaultLayout = true;
static std::string g_libraryName = "ptc";
static std::string g_rendererName = "default";
static std::vector<std::string> g_filenames;
static ImporterType g_importerType = ImporterType::NONE;
static tsd::app::Core *g_core = nullptr;
static int g_numRanks = -1;
static RemoteAppStateWindow *g_win = nullptr;
static tsd::rendering::RenderIndex *g_rIdx = nullptr;
static anari::Device g_device = nullptr;

using TSDApplication = tsd::ui::imgui::Application;
namespace tsd_ui = tsd::ui::imgui;

namespace tsd::ptc {

static void statusFunc(const void *_ctx,
    ANARIDevice device,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message)
{
  const auto *ctx = (const tsd::app::Core *)_ctx;
  if (g_rank != 0) {
    printf("[WORKER][%i][A] %s\n", g_rank, message);
    fflush(stdout);
    return;
  }

  const bool verbose = ctx->logging.verbose;

  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[ANARI][FATAL][%p] %s", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR)
    tsd::core::logError("[ANARI][ERROR] %s", source, message);
  else if (severity == ANARI_SEVERITY_WARNING)
    tsd::core::logWarning("[ANARI][WARN ][%p] %s", source, message);
  else if (verbose && severity == ANARI_SEVERITY_PERFORMANCE_WARNING)
    tsd::core::logPerfWarning("[ANARI][PERF ][%p] %s", source, message);
  else if (verbose && severity == ANARI_SEVERITY_INFO)
    tsd::core::logInfo("[ANARI][INFO ][%p] %s", source, message);
  else if (verbose && severity == ANARI_SEVERITY_DEBUG)
    tsd::core::logDebug("[ANARI][DEBUG][%p] %s", source, message);
}

static void initializeANARI()
{
  g_device = g_core->anari.loadDevice(g_libraryName);
}

static void teardownANARI()
{
  if (!g_device)
    return;
  anari::release(g_device, g_device);
  g_device = nullptr;
}

static void setupScene()
{
  // Randomize default material color //

  std::mt19937 rng;
  rng.seed(g_rank);
  std::normal_distribution<float> dist(0.2f, 0.8f);

  auto mat = g_core->tsd.ctx.defaultMaterial();
  mat->setParameter("color", float3(dist(rng), dist(rng), dist(rng)));

  // Load actual scene //

  if (g_importerType == ImporterType::NONE)
    tsd::io::generate_randomSpheres(g_core->tsd.ctx);
  else {
    for (size_t i = 0; i < g_filenames.size(); i++) {
      if (g_numRanks > 0 && (i % g_numRanks != g_rank))
        continue;
      auto root = g_core->tsd.ctx.defaultLayer()->root();
      const auto &f = g_filenames[i];
      if (g_importerType == ImporterType::PLY)
        tsd::io::import_PLY(g_core->tsd.ctx, f.c_str());
      else if (g_importerType == ImporterType::OBJ)
        tsd::io::import_OBJ(g_core->tsd.ctx, f.c_str(), root, true);
      else if (g_importerType == ImporterType::USD)
        tsd::io::import_USD(g_core->tsd.ctx, f.c_str(), root, true);
      else if (g_importerType == ImporterType::ASSIMP)
        tsd::io::import_ASSIMP(g_core->tsd.ctx, f.c_str(), root);
      else if (g_importerType == ImporterType::DLAF)
        tsd::io::import_DLAF(g_core->tsd.ctx, f.c_str(), root, true);
      else if (g_importerType == ImporterType::NBODY)
        tsd::io::import_NBODY(g_core->tsd.ctx, f.c_str());
    }
  }

  printf("[%i] %s\n",
      g_rank,
      tsd::core::objectDBInfo(g_core->tsd.ctx.objectDB()).c_str());

  // Create render index to map to a concrete ANARI objects //

  g_rIdx = g_core->anari.acquireRenderIndex(g_core->tsd.ctx, g_device);
}

///////////////////////////////////////////////////////////////////////////////
// Application definition /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

class Application : public TSDApplication
{
 public:
  Application() = default;
  ~Application() override = default;

  anari_viewer::WindowArray setupWindows() override
  {
    auto *log = new tsd::ui::imgui::Log(this, g_core);

    if (!g_core->logging.verbose)
      tsd::core::logStatus("app window running on rank '%i'", g_rank);

    auto *viewport = new tsd::ptc::DistributedViewport(
        this, g_win, g_rendererName.c_str(), "Viewport");

    g_win->fence();
    initializeANARI();
    viewport->setDevice(g_device); // creates ANARIFrame
    g_win->fence();

    setupScene();
    viewport->setManipulator(&g_core->view.manipulator);
    viewport->setWorld(g_rIdx->world(), false);
    viewport->resetView();

    anari_viewer::WindowArray windows;
    windows.emplace_back(viewport);
    windows.emplace_back(log);

    return windows;
  }

  void uiFrameStart() override
  {
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("print ImGui ini")) {
          const char *info = ImGui::SaveIniSettingsToMemory();
          printf("%s\n", info);
        }

        ImGui::EndMenu();
      }

      ImGui::EndMainMenuBar();
    }
  }

  void teardown() override
  {
    g_win->ptr()->running = false;
    g_win->fence();
    TSDApplication::teardown();
  }

  const char *getDefaultLayout() const override
  {
    return R"layout(
[Window][MainDockSpace]
Pos=0,19
Size=1600,881
Collapsed=0

[Window][Viewport]
Pos=0,19
Size=1600,638
Collapsed=0
DockId=0x00000003,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Log]
Pos=0,659
Size=1600,241
Collapsed=0
DockId=0x00000004,0

[Docking][Data]
DockSpace   ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,19 Size=1600,881 Split=Y
  DockNode  ID=0x00000003 Parent=0x80F5B4C5 SizeRef=1600,638 CentralNode=1 Selected=0xC450F867
  DockNode  ID=0x00000004 Parent=0x80F5B4C5 SizeRef=1600,241 Selected=0x139FDA3F
)layout";
  }
};

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

static void printUsage()
{
  if (g_rank != 0)
    return;

  std::cout << "./tsd::ptc [{--help|-h}]\n"
            << "   [{--verbose|-v}]\n"
            << "   [{--library|-l} <ANARI library>]\n"
            << "   [{--renderer|-r} <subtype>]\n"
            << "   [{--objFile|-obj}] <filename.obj>\n"
            << "   [{--plyFile|-ply}] <filename.ply>\n";
}

static void parseCommandLine(int argc, const char *argv[])
{
  g_core->parseCommandLine(argc, argv);

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage();
      std::exit(0);
    } else if (arg == "--noDefaultLayout")
      g_useDefaultLayout = false;
    else if (arg == "-l" || arg == "--library")
      g_libraryName = argv[++i];
    else if (arg == "-r" || arg == "--renderer")
      g_rendererName = argv[++i];
    else if (arg == "--objFile" || arg == "-obj")
      g_importerType = ImporterType::OBJ;
    else if (arg == "--assimpFile" || arg == "-assimp")
      g_importerType = ImporterType::ASSIMP;
    else if (arg == "--plyFile" || arg == "-ply")
      g_importerType = ImporterType::PLY;
    else if (arg == "--dlafFile" || arg == "-dlaf")
      g_importerType = ImporterType::DLAF;
    else if (arg == "--nbodyFile" || arg == "-nbody")
      g_importerType = ImporterType::NBODY;
    else
      g_filenames.push_back(arg);
  }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct ObjectVersions
{
  size_t camera{0};
  size_t renderer{0};
  size_t frame{0};
};

static void updateWorkerObjects(RemoteAppState &state,
    ObjectVersions &ov,
    anari::Device d,
    anari::Frame f,
    anari::Camera c,
    anari::Renderer r,
    bool force = false)
{
  if (force || ov.camera < state.camera.version) {
    anari::setParameter(d, c, "position", state.camera.position);
    anari::setParameter(d, c, "direction", state.camera.direction);
    anari::setParameter(d, c, "up", state.camera.up);
    anari::setParameter(d, c, "aspect", state.camera.aspect);
    anari::setParameter(d, c, "fovy", state.camera.fovy);
    anari::setParameter(d, c, "apertureRadius", state.camera.apertureRadius);
    anari::setParameter(d, c, "focusDistance", state.camera.focusDistance);
    anari::commitParameters(d, c);
    ov.camera = state.camera.version;
  }

  if (force || ov.renderer < state.renderer.version) {
    anari::setParameter(d, r, "background", state.renderer.background);
    anari::setParameter(
        d, r, "ambientRadiance", state.renderer.ambientRadiance);
    anari::setParameter(d, r, "ambientColor", state.renderer.ambientColor);
    anari::commitParameters(d, r);
    ov.renderer = state.renderer.version;
  }

  if (force || ov.frame < state.frame.version) {
    anari::setParameter(d, f, "size", tsd::math::uint2(state.frame.size));
    anari::commitParameters(d, f);
    ov.frame = state.frame.version;
  }
}

static void runWorker()
{
  RemoteAppState state;
  ObjectVersions ov;

  g_win->fence();

  tsd::ptc::initializeANARI();
  auto d = g_device;
  auto f = anari::newObject<anari::Frame>(d);

  g_win->fence();

  tsd::ptc::setupScene();

  auto w = g_rIdx->world();
  auto c = anari::newObject<anari::Camera>(d, "perspective");
  auto r = anari::newObject<anari::Renderer>(d, g_rendererName.c_str());

  anari::setParameter(d, f, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, f, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(d, f, "accumulation", true);
  anari::setParameter(d, f, "world", w);
  anari::setParameter(d, f, "camera", c);
  anari::setParameter(d, f, "renderer", r);

  updateWorkerObjects(state, ov, d, f, c, r, true);

  // Get dummy bounds on worker to flush commits //

  {
    tsd::math::float3 bounds[2] = {{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}};
    anariGetProperty(d,
        w,
        "bounds",
        ANARI_FLOAT32_BOX3,
        &bounds[0],
        sizeof(bounds),
        ANARI_WAIT);
  }

  while (state.running) {
    // Allow app to get the latest frame size //

    g_win->fence();

    // Get everything from the main app //

    g_win->get(0, &state, 1);
    g_win->fence();

    // Update local objects + render //

    if (state.running) {
      updateWorkerObjects(state, ov, d, f, c, r);
      anari::render(d, f);
      anari::wait(d, f);
    }
  }

  anari::release(d, f);
  anari::release(d, w);
  anari::release(d, c);
  anari::release(d, r);

  rank_printf("WORKER RENDER LOOP DONE\n");
}

} // namespace tsd::ptc

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *argv[])
{
  MPI_Init(nullptr, nullptr);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_numRanks);

  {
    std::unique_ptr<TSDApplication> app;
    std::unique_ptr<tsd::app::Core> core;
    if (g_rank == 0) {
      app = std::make_unique<tsd::ptc::Application>();
      g_core = app->appCore();
    } else {
      core = std::make_unique<tsd::app::Core>();
      g_core = core.get();
    }

    auto win = std::make_unique<RemoteAppStateWindow>();
    g_win = win.get();
    win->resize(g_rank == 0 ? 1 : 0);
    if (g_win->size() != 0)
      *g_win->ptr() = RemoteAppState();

    tsd::ptc::parseCommandLine(argc, argv);

    if (g_rank == 0)
      app->run(1600, 900, "Distributed TSD Viewer");
    else
      tsd::ptc::runWorker();

    MPI_Barrier(MPI_COMM_WORLD);

    tsd::ptc::teardownANARI();
    g_core = nullptr;
  }

  MPI_Finalize();

  return 0;
}
