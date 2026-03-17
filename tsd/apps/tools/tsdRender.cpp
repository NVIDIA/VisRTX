// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_animation
#include <tsd/animation/Animation.hpp>
#include <tsd/animation/AnimationManager.hpp>
// tsd_core
#include <tsd/core/Timer.hpp>
#include <tsd/scene/Scene.hpp>
// tsd_rendering
#include <tsd/rendering/pipeline/ImagePipeline.h>
#include <tsd/rendering/pipeline/passes/VisualizeAOVPass.h>
#include <tsd/rendering/index/RenderIndexAllLayers.hpp>
#include <tsd/rendering/view/ManipulatorToAnari.hpp>
// tsd_app
#include <tsd/app/Context.h>
// tsd_io
#include <tsd/io/serialization.hpp>
// stb_image
#include "stb_image_write.h"
// std
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

// Application state //////////////////////////////////////////////////////////

static std::unique_ptr<tsd::core::DataTree> g_stateFile;
static std::unique_ptr<tsd::scene::Scene> g_scene;
static std::unique_ptr<tsd::rendering::RenderIndexAllLayers> g_renderIndex;
static std::unique_ptr<tsd::rendering::ImagePipeline> g_renderPipeline;
static tsd::core::Timer g_timer;
static tsd::rendering::Manipulator g_manipulator;
static std::vector<tsd::rendering::CameraPose> g_cameraPoses;
static std::unique_ptr<tsd::app::Context> g_ctx;

static std::unique_ptr<tsd::animation::AnimationManager> g_animationMgr;

static tsd::core::Token g_deviceName;
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

  auto library = g_ctx->offline.renderer.libraryName;
  g_deviceName = library;

  printf("Loading ANARI device from '%s' library...", library.c_str());
  fflush(stdout);

  g_timer.start();
  g_library = anari::loadLibrary(library.c_str(), statusFunc);
  g_device = anari::newDevice(g_library, "default");
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void initTSDDataTree()
{
  printf("Initializing TSD data tree...");
  fflush(stdout);

  g_timer.start();
  g_stateFile = std::make_unique<tsd::core::DataTree>();
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void initTSDScene()
{
  printf("Initializing TSD context...");
  fflush(stdout);

  g_timer.start();
  g_scene = std::make_unique<tsd::scene::Scene>();
  g_animationMgr =
      std::make_unique<tsd::animation::AnimationManager>(g_scene.get());
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void initTSDRenderIndex()
{
  printf("Initializing TSD render index...");
  fflush(stdout);

  g_timer.start();
  g_renderIndex = std::make_unique<tsd::rendering::RenderIndexAllLayers>(
      *g_scene, g_deviceName, g_device);
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
  g_ctx->offline.loadSettings(offlineSettings);
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void populateTSDScene()
{
  printf("Populating TSD context...");
  fflush(stdout);

  g_timer.start();
  auto &root = g_stateFile->root();
  if (auto *c = root.child("context"); c != nullptr)
    tsd::io::load_Scene(*g_scene, *c);
  else
    tsd::io::load_Scene(*g_scene, root);
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
    c->foreach_child([&](tsd::core::DataNode &n) {
      tsd::rendering::CameraPose pose;
      tsd::io::nodeToCameraPose(n, pose);
      g_cameraPoses.push_back(std::move(pose));
    });
    printf("using %zu camera poses from file...", g_cameraPoses.size());
    fflush(stdout);
  } else {
    printf("from world bounds...");
    fflush(stdout);
    g_cameraPoses.push_back(g_renderIndex->computeDefaultView());
  }
  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static void setupImagePipeline()
{
  const auto frameWidth = g_ctx->offline.frame.width;
  const auto frameHeight = g_ctx->offline.frame.height;

  printf("Setting up render pipeline (%u x %u)...", frameWidth, frameHeight);
  fflush(stdout);

  g_timer.start();
  g_renderPipeline =
      std::make_unique<tsd::rendering::ImagePipeline>(frameWidth, frameHeight);

  g_camera = anari::newObject<anari::Camera>(g_device, "perspective");
  anari::setParameter(
      g_device, g_camera, "aspect", frameWidth / float(frameHeight));
  anari::setParameter(g_device, g_camera, "fovy", anari::radians(40.f));
  anari::setParameter(g_device,
      g_camera,
      "apertureRadius",
      g_ctx->offline.camera.apertureRadius);
  anari::setParameter(
      g_device, g_camera, "focusDistance", g_ctx->offline.camera.focusDistance);
  anari::commitParameters(g_device, g_camera);

  auto activeRenderer = g_ctx->offline.renderer.activeRenderer;
  auto &ro = g_ctx->offline.renderer.rendererObjects[activeRenderer];
  auto r = anari::newObject<anari::Renderer>(g_device, ro.name().c_str());
  ro.updateAllANARIParameters(g_device, r);
  anari::commitParameters(g_device, r);

  auto *arp =
      g_renderPipeline->emplace_back<tsd::rendering::AnariSceneRenderPass>(
          g_device);
  arp->setWorld(g_renderIndex->world());
  arp->setRenderer(r);
  arp->setCamera(g_camera);
  arp->setRunAsync(false);

  // Add AOV visualization pass if enabled
  if (g_ctx->offline.aov.aovType != tsd::rendering::AOVType::NONE) {
    auto *aovPass =
        g_renderPipeline->emplace_back<tsd::rendering::VisualizeAOVPass>();
    aovPass->setAOVType(g_ctx->offline.aov.aovType);
    aovPass->setDepthRange(
        g_ctx->offline.aov.depthMin, g_ctx->offline.aov.depthMax);
    aovPass->setEdgeThreshold(g_ctx->offline.aov.edgeThreshold);
    aovPass->setEdgeInvert(g_ctx->offline.aov.edgeInvert);

    // Enable necessary frame channels
    if (g_ctx->offline.aov.aovType == tsd::rendering::AOVType::ALBEDO) {
      arp->setEnableAlbedo(true);
    } else if (g_ctx->offline.aov.aovType == tsd::rendering::AOVType::NORMAL) {
      arp->setEnableNormals(true);
    } else if (g_ctx->offline.aov.aovType == tsd::rendering::AOVType::EDGES) {
      arp->setEnableIDs(true);
    }
  }

  anari::release(g_device, r);

  g_timer.end();

  printf("done (%.2f ms)\n", g_timer.milliseconds());
}

static std::string frameFilename(int i)
{
  char buf[32];
  snprintf(buf, sizeof(buf), "tsdRender_%04d.png", i);
  return buf;
}

static void renderFrames()
{
  const auto frameWidth = g_ctx->offline.frame.width;
  const auto frameHeight = g_ctx->offline.frame.height;
  const auto frameSamples = g_ctx->offline.frame.samples;

  printf("Rendering frames (%u spp)...\n", frameSamples);
  fflush(stdout);

  stbi_flip_vertically_on_write(1);

  g_timer.start();

  // Check for camera animations
  bool hasCameraAnimation = false;
  const tsd::scene::Object *animatedCamera = nullptr;
  for (auto &anim : g_animationMgr->animations()) {
    for (auto &b : anim.bindings) {
      if (b.target && b.target->type() == ANARI_CAMERA) {
        hasCameraAnimation = true;
        animatedCamera = b.target.get();
        break;
      }
    }
    if (hasCameraAnimation)
      break;
  }

  if (hasCameraAnimation) {
    const int totalFrames = g_animationMgr->getAnimationTotalFrames();

    // If no animated camera, set static pose once from saved poses
    if (!animatedCamera) {
      g_manipulator.setConfig(g_cameraPoses[0]);
      tsd::rendering::updateCameraParametersPerspective(
          g_device, g_camera, g_manipulator);
      anari::commitParameters(g_device, g_camera);
    }

    printf("...animating %d frames...\n", totalFrames);

    for (int i = 0; i < totalFrames; i++) {
      g_animationMgr->setAnimationFrame(i);

      if (animatedCamera) {
        using anari::math::float3;
        if (auto v = animatedCamera->parameterValueAs<float3>("position"))
          anari::setParameter(g_device, g_camera, "position", *v);
        if (auto v = animatedCamera->parameterValueAs<float3>("direction"))
          anari::setParameter(g_device, g_camera, "direction", *v);
        if (auto v = animatedCamera->parameterValueAs<float3>("up"))
          anari::setParameter(g_device, g_camera, "up", *v);
        if (auto v = animatedCamera->parameterValueAs<float>("fovy"))
          anari::setParameter(g_device, g_camera, "fovy", *v);
        anari::commitParameters(g_device, g_camera);
      }

      printf("...frame %d / %d...\n", i, totalFrames - 1);
      fflush(stdout);

      for (int s = 0; s < frameSamples; s++)
        g_renderPipeline->render();

      auto filename = frameFilename(i);
      stbi_write_png(filename.c_str(),
          frameWidth,
          frameHeight,
          4,
          g_renderPipeline->getColorBuffer(),
          4 * frameWidth);
    }
  } else {
    // Original camera-pose turntable behavior
    for (size_t i = 0; i < g_cameraPoses.size(); i++) {
      g_manipulator.setConfig(g_cameraPoses[i]);
      tsd::rendering::updateCameraParametersPerspective(
          g_device, g_camera, g_manipulator);
      anari::commitParameters(g_device, g_camera);

      printf("...frame %zu...\n", i);
      fflush(stdout);

      for (int s = 0; s < frameSamples; s++)
        g_renderPipeline->render();

      stbi_write_png(frameFilename(i).c_str(),
          frameWidth,
          frameHeight,
          4,
          g_renderPipeline->getColorBuffer(),
          4 * frameWidth);
    }
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
  g_scene.reset();
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

  g_ctx = std::make_unique<tsd::app::Context>();

  initTSDDataTree();
  initTSDScene();
  loadState(argv[1]);
  loadSettings();
  loadANARIDevice();
  initTSDRenderIndex();
  populateTSDScene();
  populateRenderIndex();
  setupCameraManipulator();
  setupImagePipeline();
  renderFrames();
  cleanup();

  g_ctx.reset();

  return 0;
}
