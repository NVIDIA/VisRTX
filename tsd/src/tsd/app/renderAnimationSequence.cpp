// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/app/renderAnimationSequence.h"
// tsd_app
#include "tsd/app/ANARIDeviceManager.h"
#include "tsd/app/Context.h"
// tsd_core
#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/Logging.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndexAllLayers.hpp"
#include "tsd/rendering/pipeline/ImagePipeline.h"
#include "tsd/rendering/pipeline/passes/VisualizeAOVPass.h"
// std
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>

namespace tsd::app {

void renderAnimationSequence(Context &ctx,
    const std::string &outputDir,
    const std::string &filePrefix,
    RenderSequenceCallback preFrameCallback)
{
  auto &config = ctx.offline;
  auto &scene = ctx.tsd.scene;
  auto &animMgr = ctx.tsd.animationMgr;

  // Validate renderer config //

  if (config.renderer.rendererObjects.empty()
      || config.renderer.activeRenderer < 0) {
    tsd::core::logError(
        "[renderAnimationSequence] No renderer objects configured");
    return;
  }

  auto &ro = config.renderer.rendererObjects[config.renderer.activeRenderer];
  auto libName = config.renderer.libraryName;

  tsd::core::logStatus("[renderAnimationSequence] Loading ANARI device '%s'...",
      libName.c_str());

  // Create a fresh isolated device (not shared with viewport) //

  auto library = anari::loadLibrary(libName.c_str(), anariStatusFunc, nullptr);
  if (!library) {
    tsd::core::logError(
        "[renderAnimationSequence] Failed to load ANARI library '%s'",
        libName.c_str());
    return;
  }
  auto d = anari::newDevice(library, "default");
  anari::unloadLibrary(library);
  if (!d) {
    tsd::core::logError(
        "[renderAnimationSequence] Failed to create ANARI device '%s'",
        libName.c_str());
    return;
  }
  anari::commitParameters(d, d);

  auto renderIndex =
      std::make_unique<tsd::rendering::RenderIndexAllLayers>(scene, libName, d);
  renderIndex->populate();

  // Validate camera — resolve index //

  size_t camIdx = config.camera.cameraIndex;

  // If no camera configured, find a camera-targeting binding
  if (camIdx == tsd::core::INVALID_INDEX) {
    for (const auto &anim : animMgr.animations()) {
      for (const auto &b : anim.bindings) {
        if (b.target && b.target->type() == ANARI_CAMERA) {
          camIdx = b.target->index();
          break;
        }
      }
      if (camIdx != tsd::core::INVALID_INDEX)
        break;
    }
  }

  // Last resort: use camera 0
  if (camIdx == tsd::core::INVALID_INDEX)
    camIdx = 0;

  auto cameraRef = scene.getObject<tsd::scene::Camera>(camIdx);
  if (!cameraRef) {
    tsd::core::logError(
        "[renderAnimationSequence] No camera at index %zu", camIdx);
    anari::release(d, d);
    return;
  }

  // Setup renderer //

  auto r = anari::newObject<anari::Renderer>(d, ro.subtype().c_str());
  ro.updateAllANARIParameters(d, r);
  anari::commitParameters(d, r);

  // Setup render pipeline //

  tsd::rendering::ImagePipeline pipeline;
  pipeline.setDimensions(config.frame.width, config.frame.height);

  auto *anariPass =
      pipeline.emplace_back<tsd::rendering::AnariSceneRenderPass>(d);
  anariPass->setRunAsync(false);
  anariPass->setEnableIDs(false);
  anariPass->setColorFormat(ANARI_UFIXED8_RGBA_SRGB);
  anariPass->setWorld(renderIndex->world());
  anariPass->setRenderer(r);
  anariPass->setCamera(renderIndex->camera(cameraRef->index()));

  // AOV pass //

  if (config.aov.aovType != tsd::rendering::AOVType::NONE) {
    auto *aovPass = pipeline.emplace_back<tsd::rendering::VisualizeAOVPass>();
    aovPass->setAOVType(config.aov.aovType);
    aovPass->setDepthRange(config.aov.depthMin, config.aov.depthMax);
    aovPass->setEdgeThreshold(config.aov.edgeThreshold);
    aovPass->setEdgeInvert(config.aov.edgeInvert);

    if (config.aov.aovType == tsd::rendering::AOVType::ALBEDO)
      anariPass->setEnableAlbedo(true);
    else if (config.aov.aovType == tsd::rendering::AOVType::NORMAL)
      anariPass->setEnableNormals(true);
    else if (config.aov.aovType == tsd::rendering::AOVType::EDGES
        || config.aov.aovType == tsd::rendering::AOVType::OBJECT_ID)
      anariPass->setEnableIDs(true);
    else if (config.aov.aovType == tsd::rendering::AOVType::PRIMITIVE_ID)
      anariPass->setEnablePrimitiveId(true);
    else if (config.aov.aovType == tsd::rendering::AOVType::INSTANCE_ID)
      anariPass->setEnableInstanceId(true);
  }

  auto *savePass = pipeline.emplace_back<tsd::rendering::SaveToFilePass>();
  savePass->setSingleShotMode(false);

  // Set aspect ratio on the render index's camera //

  {
    auto c = renderIndex->camera(cameraRef->index());
    anari::setParameter(d,
        c,
        "aspect",
        static_cast<float>(config.frame.width) / config.frame.height);
    anari::commitParameters(d, c);
  }

  anari::release(d, r);
  anari::release(d, d);

  // Determine frame range //

  bool hasAnimations = !animMgr.animations().empty();
  int numFrames = hasAnimations ? animMgr.getAnimationTotalFrames()
                                : config.frame.numFrames;

  auto frameStart = config.frame.renderSubset ? config.frame.startFrame : 0;
  auto frameEnd =
      config.frame.renderSubset ? config.frame.endFrame : numFrames - 1;
  auto increment = config.frame.frameIncrement;

  int savedFrame = animMgr.getAnimationFrame();

  tsd::core::logStatus(
      "[renderAnimationSequence] Rendering %d frames (%d spp) to '%s'...",
      numFrames,
      config.frame.samples,
      outputDir.c_str());

  for (int frameIndex = frameStart; frameIndex <= frameEnd;
      frameIndex += increment) {
    if (preFrameCallback) {
      if (!preFrameCallback(frameIndex, numFrames)) {
        tsd::core::logStatus(
            "[renderAnimationSequence] Aborted at frame %d", frameIndex);
        break;
      }
    }

    // Advance animation — updates TSD objects, render index commits to ANARI //
    animMgr.setAnimationFrame(frameIndex);

    // Output filename //
    std::ostringstream ss;
    ss << filePrefix << std::setfill('0') << std::setw(4) << frameIndex
       << ".png";
    std::filesystem::path filename =
        std::filesystem::path(outputDir) / ss.str();
    savePass->setFilename(filename.string());

    // Accumulate samples, save on last //
    for (int s = 0; s < config.frame.samples; ++s) {
      savePass->setEnabled(s == config.frame.samples - 1);
      pipeline.render();
    }
  }

  // Restore animation state //
  animMgr.setAnimationFrame(savedFrame);
}

} // namespace tsd::app
