// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd
#include "tsd/TSD.hpp"
// render pipeline
#include "render_pipeline/RenderPipeline.h"
// std
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>
// stb_image
#include "stb_image_write.h"

#include "AppCore.h"

// Application state //////////////////////////////////////////////////////////

static std::unique_ptr<tsd::serialization::DataTree> g_stateFile;
static std::unique_ptr<tsd::Context> g_ctx;
static std::unique_ptr<tsd::RenderIndexAllLayers> g_renderIndex;
static std::unique_ptr<tsd::RenderPipeline> g_renderPipeline;
static tsd::Timer g_timer;
static tsd::manipulators::Manipulator g_manipulator;
static std::vector<tsd::manipulators::CameraPose> g_cameraPoses;
static std::unique_ptr<tsd_viewer::AppCore> g_core;

static anari::Library g_library{nullptr};
static anari::Device g_device{nullptr};
static anari::Camera g_camera{nullptr};

// Helper functions ///////////////////////////////////////////////////////////

static void loadANARIDevice()
{
  auto statusFunc = [](const void *,
                        ANARIDevice,
                        ANARIObject,
                        ANARIDataType,
                        ANARIStatusSeverity severity,
                        ANARIStatusCode,
                        const char *message) {
    if (severity == ANARI_SEVERITY_FATAL_ERROR) {
      fprintf(stderr, "[ANARI][FATAL] %s\n", message);
      std::exit(1);
    } else if (severity == ANARI_SEVERITY_ERROR)
      fprintf(stderr, "[ANARI][ERROR] %s\n", message);
#if 0
  else if (severity == ANARI_SEVERITY_WARNING)
    fprintf(stderr, "[ANARI][WARN ] %s\n", message);
  else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING)
    fprintf(stderr, "[ANARI][PERF ] %s\n", message);
#endif
#if 0
  else if (severity == ANARI_SEVERITY_INFO)
    fprintf(stderr, "[ANARI][INFO ] %s\n", message);
  else if (severity == ANARI_SEVERITY_DEBUG)
    fprintf(stderr, "[ANARI][DEBUG] %s\n", message);
#endif
  };

  printf("Loading ANARI device from 'environment' library...");
  fflush(stdout);

  g_timer.start();
  g_library = anari::loadLibrary("environment", statusFunc);
  g_device = anari::newDevice(g_library, "default");
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void initTSDDataTree()
{
  printf("Initializing TSD data tree...");
  fflush(stdout);

  g_timer.start();
  g_stateFile = std::make_unique<tsd::serialization::DataTree>();
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void initTSDContext()
{
  printf("Initializing TSD context...");
  fflush(stdout);

  g_timer.start();
  g_ctx = std::make_unique<tsd::Context>();
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void initTSDRenderIndex()
{
  printf("Initializing TSD render index...");
  fflush(stdout);

  g_timer.start();
  g_renderIndex = std::make_unique<tsd::RenderIndexAllLayers>(*g_ctx, g_device);
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void loadState(const char *filename)
{
  printf("Loading state from '%s'...", filename);
  fflush(stdout);

  g_timer.start();
  g_stateFile->load(filename);
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void loadSettings()
{
  printf("Loading render settings...");
  fflush(stdout);

  g_timer.start();
  auto &root = g_stateFile->root();
  auto &offlineSettings = root["offlineRendering"];
  g_core->offline.loadSettings(offlineSettings);
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void populateTSDContext()
{
  printf("Populating TSD context...");
  fflush(stdout);

  g_timer.start();
  auto &root = g_stateFile->root();
  if (auto *c = root.child("context"); c != nullptr)
    tsd::load_Context(*g_ctx, *c);
  else
    tsd::load_Context(*g_ctx, root);
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void populateRenderIndex()
{
  printf("Populating TSD render index...");
  fflush(stdout);

  g_timer.start();
  g_renderIndex->populate();
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void setupCameraManipulator()
{
  printf("Setting up camera...");
  fflush(stdout);

  g_timer.start();
  auto &root = g_stateFile->root();
  if (auto *c = root.child("cameraPoses"); c != nullptr && !c->isLeaf()) {
    c->foreach_child([&](tsd::serialization::DataNode &n) {
      tsd::manipulators::CameraPose pose;
      tsd::nodeToCameraPose(n, pose);
      g_cameraPoses.push_back(std::move(pose));
    });
    printf("using %zu camera poses from file...", g_cameraPoses.size());
    fflush(stdout);
  } else {
    printf("from world bounds...");
    fflush(stdout);

    tsd::math::float3 bounds[2] = {{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}};
    anariGetProperty(g_device,
        g_renderIndex->world(),
        "bounds",
        ANARI_FLOAT32_BOX3,
        &bounds[0],
        sizeof(bounds),
        ANARI_WAIT);

    auto center = 0.5f * (bounds[0] + bounds[1]);
    auto diag = bounds[1] - bounds[0];

    tsd::manipulators::CameraPose pose;
    pose.fixedDist = 2.f * tsd::math::length(diag);
    pose.lookat = center;
    pose.azeldist = {0.f, 20.f, pose.fixedDist};
    pose.upAxis = static_cast<int>(tsd::manipulators::UpAxis::POS_Y);

    g_cameraPoses.push_back(std::move(pose));
  }
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void setupRenderPipeline()
{
  printf("Setting up render pipeline...");
  fflush(stdout);

  const auto frameWidth = g_core->offline.frame.width;
  const auto frameHeight = g_core->offline.frame.height;

  g_timer.start();
  g_renderPipeline =
      std::make_unique<tsd::RenderPipeline>(frameWidth, frameWidth);

  g_camera = anari::newObject<anari::Camera>(g_device, "perspective");
  anari::setParameter(
      g_device, g_camera, "aspect", frameWidth / float(frameWidth));
  anari::setParameter(g_device, g_camera, "fovy", anari::radians(40.f));
  anari::commitParameters(g_device, g_camera);

  auto activeRenderer = g_core->offline.renderer.activeRenderer;
  auto &ro = g_core->offline.renderer.rendererObjects[activeRenderer];
  auto r = anari::newObject<anari::Renderer>(g_device, ro.name().c_str());
  ro.updateAllANARIParameters(g_device, r);
  anari::commitParameters(g_device, r);

  auto *arp =
      g_renderPipeline->emplace_back<tsd::AnariSceneRenderPass>(g_device);
  arp->setWorld(g_renderIndex->world());
  arp->setRenderer(r);
  arp->setCamera(g_camera);

  anari::release(g_device, r);

  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void renderFrames()
{
  const auto frameWidth = g_core->offline.frame.width;
  const auto frameHeight = g_core->offline.frame.height;
  const auto frameSamples = g_core->offline.frame.samples;

  printf("Rendering frames (%u spp)...\n", frameSamples);
  fflush(stdout);

  stbi_flip_vertically_on_write(1);

  g_timer.start();

  for (size_t i = 0; i < g_cameraPoses.size(); i++) {
    const auto &pose = g_cameraPoses[i];

    g_manipulator.setConfig(pose);
    tsd::manipulators::updateCameraParametersPerspective(
        g_device, g_camera, g_manipulator);
    anari::commitParameters(g_device, g_camera);

    printf("...frame %zu...\n", i);
    fflush(stdout);

    for (int i = 0; i < frameSamples; i++)
      g_renderPipeline->render();

    std::string filename = "tsdRender_";
    if (i < 10)
      filename += "000" + std::to_string(i) + ".png";
    else if (i < 100)
      filename += "00" + std::to_string(i) + ".png";
    else if (i < 1000)
      filename += "0" + std::to_string(i) + ".png";
    else
      filename += std::to_string(i) + ".png";

    stbi_write_png(filename.c_str(),
        frameWidth,
        frameHeight,
        4,
        g_renderPipeline->getColorBuffer(),
        4 * frameWidth);
  }

  g_timer.end();

  printf("...done (%.2f ms)\n", g_timer.milliseconds());
}

static void cleanup()
{
  printf("Cleanup objects...");
  fflush(stdout);

  g_timer.start();
  g_renderPipeline.reset();
  g_renderIndex.reset();
  g_ctx.reset();
  g_stateFile.reset();
  anari::release(g_device, g_camera);
  anari::release(g_device, g_device);
  anari::unloadLibrary(g_library);
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *argv[])
{
  if (argc != 2) {
    printf("usage: %s <state_file.tsd>\n", argv[0]);
    return 1;
  }

  g_core = std::make_unique<tsd_viewer::AppCore>(nullptr);

  loadANARIDevice();
  initTSDDataTree();
  initTSDContext();
  initTSDRenderIndex();
  loadState(argv[1]);
  loadSettings();
  populateTSDContext();
  populateRenderIndex();
  setupCameraManipulator();
  setupRenderPipeline();
  renderFrames();
  cleanup();

  g_core.reset();

  return 0;
}
