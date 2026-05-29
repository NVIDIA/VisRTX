// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderShot.h"

#include "tsd/app/ANARIDeviceManager.h"
#include "tsd/core/Logging.hpp"
#include "tsd/rendering/index/RenderIndexAllLayers.hpp"
#include "tsd/rendering/pipeline/ImagePipeline.h"
#include "tsd/rendering/pipeline/passes/AnariSceneRenderPass.h"
#include "tsd/rendering/pipeline/passes/SaveToFilePass.h"

#include <filesystem>
#include <iomanip>
#include <sstream>

namespace tsd::scivis_studio {


namespace {

anari::Device loadFirstAvailableDevice(
    tsd::app::ANARIDeviceManager &deviceManager, std::string &libName)
{
  auto tryLoad = [&](const std::string &name) {
    return deviceManager.loadDevice(name);
  };

  if (auto device = tryLoad(libName))
    return device;

  if (!libName.empty() && libName != "{none}") {
    tsd::core::logWarning(
        "[SciVisStudio] Failed to load ANARI device '%s'; falling back to a "
        "default device",
        libName.c_str());
  }

  for (const auto &fallback : deviceManager.libraryList()) {
    if (fallback == libName)
      continue;
    if (auto device = tryLoad(fallback)) {
      libName = fallback;
      return device;
    }
  }

  libName.clear();
  return nullptr;
}

} // namespace

bool renderActiveShotToFrames(
    ProjectContext &projectContext, RenderShotProgress *progress)
{
  auto *ctx = projectContext.appContext();
  auto *shot = project::activeShot(projectContext.project());
  if (!ctx || !shot)
    return false;

  if (!projectContext.project().isSaved()) {
    tsd::core::logError("[SciVisStudio] Cannot render an unsaved project");
    return false;
  }

  auto *cameraObject = projectContext.resolveShotCamera(*shot);
  if (!cameraObject || cameraObject->type() != ANARI_CAMERA) {
    tsd::core::logError("[SciVisStudio] Active shot camera is missing");
    return false;
  }

  const auto outputDirectory =
      projectContext.project().projectDirectory / "renders" / shot->id;
  std::error_code ec;
  std::filesystem::create_directories(outputDirectory, ec);
  if (ec) {
    tsd::core::logError("[SciVisStudio] Failed to create render directory '%s'",
        outputDirectory.string().c_str());
    return false;
  }

  auto libName = shot->renderSettings.rendererLibrary;
  auto device = loadFirstAvailableDevice(ctx->anari, libName);
  if (!device) {
    tsd::core::logError(
        "[SciVisStudio] Failed to load an ANARI device for shot rendering");
    return false;
  }

  projectContext.applyActiveShot();

  auto *renderIndex = ctx->tsd.scene.updateDelegate()
                          .emplace<tsd::rendering::RenderIndexAllLayers>(
                              ctx->tsd.scene, libName, device);
  renderIndex->populate();

  const auto rendererIndex = shot->renderSettings.rendererObjectIndex;
  auto rendererObject = ctx->tsd.scene.getObject(ANARI_RENDERER, rendererIndex);
  if (!rendererObject || rendererObject->rendererDeviceName() != libName) {
    tsd::core::logError(
        "[SciVisStudio] Renderer object index %zu is unavailable for ANARI "
        "device '%s'",
        rendererIndex,
        libName.c_str());
    ctx->tsd.scene.updateDelegate().erase(renderIndex);
    anari::release(device, device);
    return false;
  }

  auto renderer = renderIndex->renderer(rendererIndex);
  if (!renderer) {
    tsd::core::logError(
        "[SciVisStudio] Failed to resolve renderer object index %zu",
        rendererIndex);
    ctx->tsd.scene.updateDelegate().erase(renderIndex);
    anari::release(device, device);
    return false;
  }

  tsd::rendering::ImagePipeline pipeline;
  pipeline.setDimensions(
      shot->renderSettings.width, shot->renderSettings.height);
  auto *anariPass =
      pipeline.emplace_back<tsd::rendering::AnariSceneRenderPass>(device);
  anariPass->setRunAsync(false);
  anariPass->setColorFormat(ANARI_UFIXED8_RGBA_SRGB);
  anariPass->setWorld(renderIndex->world());
  anariPass->setRenderer(renderer);
  anariPass->setCamera(renderIndex->camera(shot->camera.objectIndex));

  auto *savePass = pipeline.emplace_back<tsd::rendering::SaveToFilePass>();
  savePass->setSingleShotMode(false);

  if (auto camera = renderIndex->camera(shot->camera.objectIndex)) {
    anari::setParameter(device,
        camera,
        "aspect",
        static_cast<float>(shot->renderSettings.width)
            / static_cast<float>(shot->renderSettings.height));
    anari::commitParameters(device, camera);
  }

  const int savedFrame = shot->currentFrame;
  const bool savedPlaying = shot->playing;
  const int totalFrames = std::max(1, shot->frameCount);
  const auto prefix = shot->renderSettings.outputFilePrefix.empty()
      ? shot->id
      : shot->renderSettings.outputFilePrefix;
  shot->playing = false;
  projectContext.syncAnimationManagerToActiveShot();

  tsd::core::logStatus("[SciVisStudio] Rendering %d frames to '%s'",
      totalFrames,
      outputDirectory.string().c_str());

  bool completed = true;
  for (int frame = 0; frame < totalFrames; ++frame) {
    if (progress && progress->onFrame
        && !progress->onFrame(frame, totalFrames)) {
      tsd::core::logStatus(
          "[SciVisStudio] Shot render canceled before frame %d/%d",
          frame,
          totalFrames);
      completed = false;
      break;
    }

    ctx->tsd.animationMgr.setAnimationFrame(frame);

    std::ostringstream ss;
    ss << prefix << '_' << std::setfill('0') << std::setw(4) << frame << ".png";
    savePass->setFilename((outputDirectory / ss.str()).string());

    for (uint32_t sample = 0; sample < shot->renderSettings.samples; ++sample) {
      savePass->setEnabled(sample + 1 == shot->renderSettings.samples);
      pipeline.render();
    }
  }

  shot->currentFrame = savedFrame;
  shot->playing = savedPlaying;
  projectContext.syncAnimationManagerToActiveShot();
  projectContext.applyActiveShot();

  ctx->tsd.scene.updateDelegate().erase(renderIndex);
  anari::release(device, device);

  return completed;
}

} // namespace tsd::scivis_studio
