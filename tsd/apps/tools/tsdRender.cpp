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
// stb_image
#include "stb_image_write.h"

// Application state //////////////////////////////////////////////////////////

static std::unique_ptr<tsd::serialization::DataTree> g_stateFile;
static std::unique_ptr<tsd::Context> g_ctx;
static std::unique_ptr<tsd::RenderIndexAllLayers> g_renderIndex;
static std::unique_ptr<tsd::RenderPipeline> g_renderPipeline;
static tsd::uint2 g_imageSize = {1200, 800};
static tsd::Timer g_timer;
static tsd::manipulators::Orbit g_manipulator;

static anari::Library g_library{nullptr};
static anari::Device g_device{nullptr};

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
  g_renderIndex =
      std::make_unique<tsd::RenderIndexAllLayers>(*g_ctx, g_device);
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
    tsd::manipulators::CameraPose pose;
    tsd::nodeToCameraPose(*c->child(0), pose);
    printf("using camera pose '%s'...", pose.name.c_str());
    fflush(stdout);
    g_manipulator.setConfig(pose);
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
    g_manipulator.setConfig(
        center, 0.5f * tsd::math::length(diag), {0.f, 20.f});
  }
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void setupRenderPipeline()
{
  printf("Setting up render pipeline...");
  fflush(stdout);

  g_timer.start();
  g_renderPipeline =
      std::make_unique<tsd::RenderPipeline>(g_imageSize.x, g_imageSize.y);

  auto camera = anari::newObject<anari::Camera>(g_device, "perspective");
  anari::setParameter(g_device, camera, "position", g_manipulator.eye());
  anari::setParameter(g_device, camera, "direction", g_manipulator.dir());
  anari::setParameter(g_device, camera, "up", g_manipulator.up());
  anari::setParameter(
      g_device, camera, "aspect", g_imageSize.x / float(g_imageSize.y));
  anari::setParameter(g_device, camera, "fovy", anari::radians(40.f));
  anari::commitParameters(g_device, camera);

  auto renderer = anari::newObject<anari::Renderer>(g_device, "default");
  anari::setParameter(g_device, renderer, "ambientRadiance", 0.25f);
  anari::commitParameters(g_device, renderer);

  auto *arp = g_renderPipeline->emplace_back<tsd::AnariSceneRenderPass>(g_device);
  arp->setWorld(g_renderIndex->world());
  arp->setRenderer(renderer);
  arp->setCamera(camera);

  anari::release(g_device, camera);
  anari::release(g_device, renderer);

  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void renderFrame()
{
  printf("Rendering frame (64 spp)...");
  fflush(stdout);

  g_timer.start();
  for (int i = 0; i < 64; i++)
    g_renderPipeline->render();
  stbi_flip_vertically_on_write(1);
  stbi_write_png("tsdRender.png",
      g_imageSize.x,
      g_imageSize.y,
      4,
      g_renderPipeline->getColorBuffer(),
      4 * g_imageSize.x);
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
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

  loadANARIDevice();
  initTSDDataTree();
  initTSDContext();
  initTSDRenderIndex();
  loadState(argv[1]);
  populateTSDContext();
  populateRenderIndex();
  setupCameraManipulator();
  setupRenderPipeline();
  renderFrame();
  cleanup();

  return 0;
}
