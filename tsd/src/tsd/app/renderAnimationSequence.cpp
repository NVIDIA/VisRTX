// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/app/renderAnimationSequence.h"
// tsd_app
#include "tsd/app/Core.h"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndex.hpp"
#include "tsd/rendering/pipeline/RenderPipeline.h"
#include "tsd/rendering/pipeline/passes/VisualizeAOVPass.h"
// std
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>

namespace tsd::app {

// Helper types ///////////////////////////////////////////////////////////////

struct OfflineRenderRig
{
  std::unique_ptr<tsd::rendering::RenderPipeline> pipeline;
  tsd::rendering::AnariSceneRenderPass *anariPass{nullptr};
  tsd::rendering::SaveToFilePass *saveToFilePass{nullptr};

  tsd::rendering::RenderIndex *renderIndex{nullptr};

  tsd::core::CameraRef camera;

  int numFrames{0};
};

// Helper functions ///////////////////////////////////////////////////////////

static OfflineRenderRig setupRig(tsd::app::Core &core)
{
  OfflineRenderRig rig;

  // Setup render index //

  auto &config = core.offline;
  auto d = core.anari.loadDevice(config.renderer.libraryName.c_str());

  auto &scene = core.tsd.scene;
  rig.renderIndex = core.anari.acquireRenderIndex(scene, d);

  // Setup camera //

  rig.camera = scene.getObject<tsd::core::Camera>(config.camera.cameraIndex);

  if (!rig.camera) {
    tsd::core::logError(
        "[renderAnimationSequence] No camera objects configured");
    anari::release(d, d);
    return {};
  }

  auto c = anari::newObject<anari::Camera>(d, "perspective");

  // Setup renderer //

  if (config.renderer.rendererObjects.empty()
      || config.renderer.activeRenderer < 0) {
    tsd::core::logError(
        "[renderAnimationSequence] No renderer objects configured");
    anari::release(d, c);
    anari::release(d, d);
    return {};
  }

  auto &ro = config.renderer.rendererObjects[config.renderer.activeRenderer];
  auto r = anari::newObject<anari::Renderer>(d, ro.subtype().c_str());
  ro.updateAllANARIParameters(d, r);
  anari::commitParameters(d, r);

  // Create pipeline stages //

  rig.pipeline = std::make_unique<tsd::rendering::RenderPipeline>();

  rig.anariPass =
      rig.pipeline->emplace_back<tsd::rendering::AnariSceneRenderPass>(d);
  rig.saveToFilePass =
      rig.pipeline->emplace_back<tsd::rendering::SaveToFilePass>();

  // Configure pipeline stages //

  rig.pipeline->setDimensions(config.frame.width, config.frame.height);

  rig.anariPass->setRunAsync(false);
  rig.anariPass->setEnableIDs(false);
  rig.anariPass->setColorFormat(ANARI_UFIXED8_RGBA_SRGB);
  rig.anariPass->setWorld(rig.renderIndex->world());
  rig.anariPass->setRenderer(r);
  rig.anariPass->setCamera(c);

  // Add AOV visualization pass if enabled
  if (config.aov.aovType != tsd::rendering::AOVType::NONE) {
    auto *aovPass =
        rig.pipeline->emplace_back<tsd::rendering::VisualizeAOVPass>();
    aovPass->setAOVType(config.aov.aovType);
    aovPass->setDepthRange(config.aov.depthMin, config.aov.depthMax);
    aovPass->setEdgeThreshold(config.aov.edgeThreshold);
    aovPass->setEdgeInvert(config.aov.edgeInvert);

    // Enable necessary frame channels
    if (config.aov.aovType == tsd::rendering::AOVType::ALBEDO) {
      rig.anariPass->setEnableAlbedo(true);
    } else if (config.aov.aovType == tsd::rendering::AOVType::NORMAL) {
      rig.anariPass->setEnableNormals(true);
    } else if (config.aov.aovType == tsd::rendering::AOVType::EDGES) {
      rig.anariPass->setEnableIDs(true);
    }
  }

  rig.saveToFilePass->setEnabled(true);
  rig.saveToFilePass->setSingleShotMode(false);

  // Cleanup //

  anari::release(d, r);
  anari::release(d, c);
  anari::release(d, d);

  return std::move(rig);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void renderAnimationSequence(Core &core,
    const std::string &outputDir,
    const std::string &filePrefix,
    RenderSequenceCallback preFrameCallback)
{
  auto rp = setupRig(core);

  if (rp.pipeline.get() == nullptr) {
    tsd::core::logError(
        "[renderAnimationSequence]"
        " Aborting render sequence due to setup error");
    return;
  }

  auto &scene = core.tsd.scene;
  float originalTime = scene.getAnimationTime();

  auto d = rp.anariPass->getDevice();
  auto c = rp.anariPass->getCamera();

  auto &config = core.offline.frame;
  auto start = config.renderSubset ? config.startFrame : 0;
  auto end = config.renderSubset ? config.endFrame : config.numFrames - 1;
  auto increment = config.frameIncrement;

  for (int frameIndex = start; frameIndex <= end; frameIndex += increment) {
    if (preFrameCallback) {
      if (!preFrameCallback(frameIndex, config.numFrames)) {
        tsd::core::logStatus(
            "[renderAnimationSequence] Aborting render sequence at frame %d",
            frameIndex);
        break;
      }
    }

    // Set scene time for this frame //

    float time = static_cast<float>(frameIndex) / (config.numFrames - 1);
    scene.setAnimationTime(time);

    // Update camera (in case it is animated) //

    rp.camera->updateAllANARIParameters(d, c);
    anari::setParameter(d,
        c,
        "aspect",
        static_cast<float>(core.offline.frame.width)
            / core.offline.frame.height);
    anari::commitParameters(d, c);

    // Setup output file pass //

    std::ostringstream ss;
    ss << filePrefix << std::setfill('0') << std::setw(4) << frameIndex
       << ".png";
    std::filesystem::path filename =
        std::filesystem::path(outputDir) / ss.str();

    rp.saveToFilePass->setFilename(filename.string());

    // Render the frame //

    for (int i = 0; i < core.offline.frame.samples; ++i) {
      rp.saveToFilePass->setEnabled(i == core.offline.frame.samples - 1);
      rp.pipeline->render();
    }
  }

  // Cleanup //

  scene.setAnimationTime(originalTime);
  core.anari.releaseRenderIndex(rp.anariPass->getDevice());
}

} // namespace tsd::app
