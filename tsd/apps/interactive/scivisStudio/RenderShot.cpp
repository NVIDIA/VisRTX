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

bool renderActiveShotToFrames(
    ProjectContext &projectContext, RenderShotProgress *progress)
{
  auto *ctx = projectContext.appContext();
  auto *shot = activeShot(projectContext.project());
  if (!ctx || !shot)
    return false;

  if (!projectContext.project().isSaved()) {
    tsd::core::logError("[SciVisStudio] Cannot render an unsaved project");
    return false;
  }

  auto *cameraObject = projectContext.resolve(shot->camera);
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
  auto subtype = shot->renderSettings.rendererSubtype.empty()
      ? std::string("default")
      : shot->renderSettings.rendererSubtype;

  auto library =
      anari::loadLibrary(libName.c_str(), tsd::app::anariStatusFunc, nullptr);
  if (!library) {
    tsd::core::logError(
        "[SciVisStudio] Failed to load ANARI library '%s'", libName.c_str());
    return false;
  }

  auto device = anari::newDevice(library, "default");
  anari::unloadLibrary(library);
  if (!device) {
    tsd::core::logError(
        "[SciVisStudio] Failed to create ANARI device '%s'", libName.c_str());
    return false;
  }
  anari::commitParameters(device, device);

  auto *renderIndex =
      ctx->tsd.scene.updateDelegate()
          .emplace<tsd::rendering::RenderIndexAllLayers>(
              ctx->tsd.scene, libName, device);
  renderIndex->populate();

  auto renderer = anari::newObject<anari::Renderer>(device, subtype.c_str());
  anari::commitParameters(device, renderer);

  tsd::rendering::ImagePipeline pipeline;
  pipeline.setDimensions(shot->renderSettings.width, shot->renderSettings.height);
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
  const int totalFrames = std::max(1, shot->frameCount);
  const auto prefix = shot->renderSettings.outputFilePrefix.empty()
      ? shot->id
      : shot->renderSettings.outputFilePrefix;

  tsd::core::logStatus("[SciVisStudio] Rendering %d frames to '%s'",
      totalFrames,
      outputDirectory.string().c_str());

  for (int frame = 0; frame < totalFrames; ++frame) {
    if (progress && progress->onFrame && !progress->onFrame(frame, totalFrames))
      break;

    shot->currentFrame = frame;
    projectContext.applyActiveShot();

    std::ostringstream ss;
    ss << prefix << '_' << std::setfill('0') << std::setw(4) << frame
       << ".png";
    savePass->setFilename((outputDirectory / ss.str()).string());

    for (uint32_t sample = 0; sample < shot->renderSettings.samples; ++sample) {
      savePass->setEnabled(sample + 1 == shot->renderSettings.samples);
      pipeline.render();
    }
  }

  shot->currentFrame = savedFrame;
  projectContext.applyActiveShot();

  ctx->tsd.scene.updateDelegate().erase(renderIndex);
  anari::release(device, renderer);
  anari::release(device, device);

  return true;
}

} // namespace tsd::scivis_studio
