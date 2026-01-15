// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CameraPoses.h"
// tsd_app
#include "tsd/app/Core.h"
#include "tsd/app/renderAnimationSequence.h"
// tsd_ui_imgui
#include "tsd/ui/imgui/windows/Viewport.h"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndex.hpp"
#include "tsd/rendering/pipeline/RenderPipeline.h"
#include "tsd/rendering/view/ManipulatorToAnari.hpp"
// imgui
#include <misc/cpp/imgui_stdlib.h>
// std
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace tsd::ui::imgui {

CameraPoses::CameraPoses(Application *app, Viewport *viewport, const char *name)
    : Window(app, name), m_viewport(viewport)
{}

void CameraPoses::buildUI()
{
  // Check if we have a new pose from rendering thread
  if (m_hasNewPose.load()) {
    std::lock_guard<std::mutex> lock(m_poseMutex);
    appCore()->setCameraPose(m_currentPose);
    m_hasNewPose.store(false);
  }

  ImGui::Text("Add:");
  ImGui::SameLine();

  if (ImGui::Button("current view"))
    appCore()->addCurrentViewToCameraPoses();
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("insert new view using the current camera view");

  ImGui::SameLine();
  if (ImGui::Button("turntable views"))
    ImGui::OpenPopup("CameraPoses_turntablePopupMenu");

  ImGui::SameLine();
  if (ImGui::Button("add camera object")) {
    if (m_viewport)
      m_viewport->addCameraObjectFromCurrentView();
    else
      tsd::core::logWarning("Viewport window not found");
  }

  ImGui::SameLine();
  ImGui::Text(" | ");
  ImGui::SameLine();

  if (ImGui::Button("clear"))
    ImGui::OpenPopup("CameraPoses_confirmPopupMenu");

  ImGui::Separator();

  int i = 0;
  int toRemove = -1;

  const ImGuiTableFlags flags = ImGuiTableFlags_RowBg
      | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV;

  if (ImGui::BeginTable("camera poses", 4, flags)) {
    for (auto &p : appCore()->view.poses) {
      ImGui::PushID(&p);

      ImGui::TableNextRow();

      ImGui::TableSetColumnIndex(0);
      ImGui::SetNextItemWidth(-1.f);
      ImGui::InputText("##", &p.name);

      ImGui::TableSetColumnIndex(1);
      if (ImGui::Button(">"))
        appCore()->setCameraPose(p);
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("set as current view");

      ImGui::TableSetColumnIndex(2);
      if (ImGui::Button("+")) {
        appCore()->updateExistingCameraPoseFromView(p);
        tsd::core::logStatus("camera pose '%s' updated", p.name.c_str());
      }
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("update this pose from current view");

      ImGui::TableSetColumnIndex(3);
      if (ImGui::Button("-"))
        toRemove = i;
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("delete this pose");
      ImGui::PopID();
      i++;
    }

    ImGui::EndTable();
  }

  if (toRemove >= 0)
    appCore()->view.poses.erase(appCore()->view.poses.begin() + toRemove);

  ImGui::Separator();
  buildUI_interpolationControls();

  buildUI_turntablePopupMenu();
  buildUI_confirmPopupMenu();
}

void CameraPoses::buildUI_turntablePopupMenu()
{
  if (ImGui::BeginPopup("CameraPoses_turntablePopupMenu")) {
    ImGui::InputFloat3("azimuths", &m_turntableAzimuths.x, "%.3f");
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("{min, max, step size}");

    ImGui::InputFloat3("elevations", &m_turntableElevations.x, "%.3f");
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("{min, max, step size}");

    ImGui::InputFloat3("center", &m_turntableCenter.x, "%.3f");
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("view center");

    ImGui::InputFloat("distance", &m_turntableDistance, 0.01f, 0.1f, "%.3f");
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("view distance from center");

    if (ImGui::Button("ok")) {
      appCore()->addTurntableCameraPoses(m_turntableAzimuths,
          m_turntableElevations,
          m_turntableCenter,
          m_turntableDistance);
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("cancel"))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
}

void CameraPoses::buildUI_confirmPopupMenu()
{
  if (ImGui::BeginPopup("CameraPoses_confirmPopupMenu")) {
    ImGui::Text("are you sure?");
    if (ImGui::Button("yes")) {
      appCore()->removeAllPoses();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("cancel"))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
}

void CameraPoses::buildUI_interpolationControls()
{
  ImGui::Text("Camera Path Animation:");
  ImGui::Indent(INDENT_AMOUNT);

  const bool hasPoses = appCore()->view.poses.size() >= 2;

  ImGui::BeginDisabled(!hasPoses || m_isRendering);

  auto &pathSettings = appCore()->view.pathSettings;

  // Interpolation type selector
  const char *interpTypes[] = {
      "Linear", "Smooth Step", "Catmull-Rom Spline", "Cubic Bezier"};
  int currentType = static_cast<int>(pathSettings.type);
  if (ImGui::Combo("type", &currentType, interpTypes, 4)) {
    pathSettings.type =
        static_cast<tsd::rendering::CameraPathInterpolationType>(currentType);
  }
  if (ImGui::IsItemHovered() && !m_isRendering) {
    switch (pathSettings.type) {
    case tsd::rendering::CameraPathInterpolationType::LINEAR:
      ImGui::SetTooltip("Simple linear interpolation (straight paths)");
      break;
    case tsd::rendering::CameraPathInterpolationType::SMOOTH_STEP:
      ImGui::SetTooltip("Smooth acceleration/deceleration at keyframes");
      break;
    case tsd::rendering::CameraPathInterpolationType::CATMULL_ROM:
      ImGui::SetTooltip("Smooth spline passing through all keyframes");
      break;
    case tsd::rendering::CameraPathInterpolationType::CUBIC_BEZIER:
      ImGui::SetTooltip("Smooth cubic interpolation with tension control");
      break;
    }
  }

  // Tension slider for spline-based interpolation
  if (pathSettings.type
          == tsd::rendering::CameraPathInterpolationType::CATMULL_ROM
      || pathSettings.type
          == tsd::rendering::CameraPathInterpolationType::CUBIC_BEZIER) {
    ImGui::DragFloat(
        "tension", &pathSettings.tension, 0.01f, 0.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Controls curve tightness (0=loose, 1=tight)");
    }
  }

  ImGui::DragInt(
      "frames per pose", &pathSettings.framesPerSegment, 1.0f, 1, 1000);
  if (ImGui::IsItemHovered() && !m_isRendering) {
    if (hasPoses) {
      ImGui::SetTooltip(
          "Number of frames per pose (controls total frame count)");
    } else {
      ImGui::SetTooltip("Add at least 2 poses to enable interpolation");
    }
  }

  ImGui::DragInt("fps", &pathSettings.framesPerSecond, 1.0f, 1, 240);
  if (ImGui::IsItemHovered() && !m_isRendering) {
    ImGui::SetTooltip("Frames per second for the camera animation timeline");
  }

  ImGui::Checkbox("update viewport", &m_updateViewport);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Update the main viewport camera while rendering (live preview)");
  }

  ImGui::EndDisabled(); // End disable block for controls

  ImGui::Separator();

  // Renderer configuration notice
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
  ImGui::TextWrapped(
      "Note: Rendering uses Offline Render Settings. "
      "To change output folder, file prefix, and renderer: File / App Settings / Offline Render Settings");
  ImGui::PopStyleColor();

  // Render or Cancel button
  if (!m_isRendering) {
    ImGui::BeginDisabled(!hasPoses); // Only disable render button if no poses
    if (ImGui::Button("Render")) {
      m_cancelRequested = false;
      renderInterpolatedPath();
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered() && hasPoses) {
      const int numPoses = static_cast<int>(appCore()->view.poses.size());
      const int totalFrames = numPoses * pathSettings.framesPerSegment + 1;
      const float durationSeconds = (totalFrames - 1)
          / static_cast<float>(std::max(1, pathSettings.framesPerSecond));
      ImGui::SetTooltip(
          "Render %d frames (%d poses × %d frames per pose, %.2fs @ %d fps)",
          totalFrames,
          numPoses,
          pathSettings.framesPerSegment,
          durationSeconds,
          pathSettings.framesPerSecond);
    }
  } else {
    // Cancel button is always enabled during rendering
    if (ImGui::Button("Cancel")) {
      m_cancelRequested = true;
      tsd::core::logStatus("[CameraPoses] Cancellation requested...");
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Cancel the current rendering");
    }
  }

  // Show rendering progress
  if (m_isRendering) {
    // Update timer
    m_renderTimer.end();

    // Check if rendering is complete
    if (m_renderFuture.valid()
        && m_renderFuture.wait_for(std::chrono::milliseconds(0))
            == std::future_status::ready) {
      m_renderFuture.get();
      m_isRendering = false;

      if (m_cancelRequested) {
        tsd::core::logStatus("[CameraPoses] Rendering cancelled (%.2f seconds)",
            m_renderTimer.seconds());
      } else {
        tsd::core::logStatus("[CameraPoses] Rendering complete (%.2f seconds)",
            m_renderTimer.seconds());
      }
    } else {
      // Calculate progress
      float progress = 0.0f;
      if (m_totalFrames > 0) {
        progress = static_cast<float>(m_currentFrame)
            / static_cast<float>(m_totalFrames);
      }

      // Show progress bar with percentage
      char progressText[64];
      snprintf(progressText,
          sizeof(progressText),
          "%d / %d frames (%.1f%%)",
          m_currentFrame,
          m_totalFrames,
          progress * 100.0f);
      ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), progressText);
    }
  }

  ImGui::Unindent(INDENT_AMOUNT);
}

void CameraPoses::renderInterpolatedPath()
{
  auto *core = appCore();
  const auto &poses = core->view.poses;

  if (poses.size() < 2) {
    tsd::core::logWarning(
        "[CameraPoses] Need at least 2 poses for interpolation");
    return;
  }

  // NOTE: Camera path rendering will use the viewport's active renderer
  // This ensures exported frames match what you see in the viewport

  // Auto-initialize offline renderer settings if not configured
  if (core->offline.renderer.rendererObjects.empty()
      || core->offline.renderer.libraryName.empty()) {
    if (!core->commandLine.libraryList.empty()) {
      tsd::core::logStatus(
          "[CameraPoses] Initializing offline renderer from viewport device");
      core->setOfflineRenderingLibrary(core->commandLine.libraryList[0]);
    } else {
      tsd::core::logError(
          "[CameraPoses] No ANARI library available. Cannot render.");
      return;
    }
  }

  // Log what renderer we're using with details
  if (core->offline.renderer.activeRenderer >= 0
      && core->offline.renderer.activeRenderer
          < static_cast<int>(core->offline.renderer.rendererObjects.size())) {
    const auto &rendererObj =
        core->offline.renderer
            .rendererObjects[core->offline.renderer.activeRenderer];
    tsd::core::logStatus("[CameraPoses] Rendering camera path with:");
    tsd::core::logStatus(
        "  Device: '%s'", core->offline.renderer.libraryName.c_str());
    tsd::core::logStatus("  Renderer: '%s' (index %d)",
        rendererObj.subtype().c_str(),
        core->offline.renderer.activeRenderer);
  } else {
    tsd::core::logStatus(
        "[CameraPoses] Rendering with: device='%s', renderer index %d",
        core->offline.renderer.libraryName.c_str(),
        core->offline.renderer.activeRenderer);
  }

  // Create output directory from offline settings
  const std::string &outputDirectory = core->offline.output.outputDirectory;
  const std::string &filePrefix = core->offline.output.filePrefix;

  if (outputDirectory.empty()) {
    tsd::core::logError(
        "[CameraPoses] Output directory not set. Please configure in: File → "
        "App Settings → Offline Render Settings");
    return;
  }

  std::filesystem::path outputPath(outputDirectory);
  std::filesystem::create_directories(outputPath);

  std::vector<tsd::rendering::CameraPose> samples;
  tsd::rendering::buildCameraPathSamples(
      poses, core->view.pathSettings, samples);

  if (samples.empty()) {
    tsd::core::logWarning("[CameraPoses] No camera path samples generated");
    return;
  }

  core->updateCameraPathAnimation();

  m_isRendering = true;
  m_renderTimer.start();
  m_currentFrame = 0;
  m_totalFrames = static_cast<int>(samples.size());

  tsd::core::logStatus(
      "[CameraPoses] Starting interpolated path rendering: %d frames to '%s' "
      "(prefix='%s')",
      m_totalFrames,
      outputDirectory.c_str(),
      filePrefix.c_str());

  // Capture settings by value
  const std::string capturedOutputDirectory = outputDirectory;
  const std::string capturedFilePrefix = filePrefix;
  const bool updateViewport = m_updateViewport;
  std::vector<tsd::rendering::CameraPose> samplesCopy = samples;

  // Calculate total frames for capture
  const int capturedTotalFrames = m_totalFrames;

  // Launch rendering in background
  m_renderFuture = std::async(std::launch::async,
      [this,
          core,
          samplesCopy,
          capturedOutputDirectory,
          capturedFilePrefix,
          updateViewport,
          capturedTotalFrames]() {
        // Setup render pipeline
        auto &config = core->offline;
        auto d = core->anari.loadDevice(config.renderer.libraryName.c_str());
        if (!d) {
          tsd::core::logError("[CameraPoses] Failed to load ANARI device");
          return;
        }

        auto &scene = core->tsd.scene;
        auto *renderIndex = core->anari.acquireRenderIndex(scene, d);
        if (!renderIndex) {
          tsd::core::logError("[CameraPoses] Failed to acquire render index");
          anari::release(d, d);
          return;
        }

        // Create renderer
        if (config.renderer.rendererObjects.empty()
            || config.renderer.activeRenderer < 0
            || config.renderer.activeRenderer
                >= static_cast<int>(config.renderer.rendererObjects.size())) {
          tsd::core::logError("[CameraPoses] No renderer configured");
          anari::release(d, d);
          return;
        }

        // Log renderer details at creation time
        tsd::core::logStatus(
            "[CameraPoses] Creating renderer in background thread:");
        tsd::core::logStatus(
            "  activeRenderer index: %d", config.renderer.activeRenderer);
        tsd::core::logStatus("  rendererObjects.size(): %zu",
            config.renderer.rendererObjects.size());

        auto &ro =
            config.renderer.rendererObjects[config.renderer.activeRenderer];

        tsd::core::logStatus("  Renderer subtype: '%s'", ro.subtype().c_str());
        tsd::core::logStatus("  Renderer name: '%s'", ro.name().c_str());

        auto r = anari::newObject<anari::Renderer>(d, ro.subtype().c_str());
        tsd::core::logStatus("  Created ANARI renderer object");

        ro.updateAllANARIParameters(d, r);
        tsd::core::logStatus("  Updated renderer parameters");

        anari::commitParameters(d, r);
        tsd::core::logStatus("  Committed renderer parameters");

        // Create camera
        auto c = anari::newObject<anari::Camera>(d, "perspective");
        anari::setParameter(d,
            c,
            "aspect",
            static_cast<float>(config.frame.width) / config.frame.height);
        anari::setParameter(d, c, "fovy", anari::radians(40.f));
        anari::commitParameters(d, c);

        // Create render pipeline
        auto pipeline = std::make_unique<tsd::rendering::RenderPipeline>(
            config.frame.width, config.frame.height);

        auto *anariPass =
            pipeline->emplace_back<tsd::rendering::AnariSceneRenderPass>(d);
        anariPass->setRunAsync(false);
        anariPass->setWorld(renderIndex->world());
        anariPass->setRenderer(r);
        anariPass->setCamera(c);

        // Add AOV visualization pass if enabled
        if (config.aov.aovType != tsd::rendering::AOVType::NONE) {
          auto *aovPass =
              pipeline->emplace_back<tsd::rendering::VisualizeAOVPass>();
          aovPass->setAOVType(config.aov.aovType);
          aovPass->setDepthRange(config.aov.depthMin, config.aov.depthMax);
          aovPass->setEdgeThreshold(config.aov.edgeThreshold);
          aovPass->setEdgeInvert(config.aov.edgeInvert);

          // Enable necessary frame channels
          if (config.aov.aovType == tsd::rendering::AOVType::ALBEDO) {
            anariPass->setEnableAlbedo(true);
          } else if (config.aov.aovType == tsd::rendering::AOVType::NORMAL) {
            anariPass->setEnableNormals(true);
          } else if (config.aov.aovType == tsd::rendering::AOVType::EDGES) {
            anariPass->setEnableIDs(true);
          }
        }

        auto *savePass =
            pipeline->emplace_back<tsd::rendering::SaveToFilePass>();
        savePass->setSingleShotMode(false);

        // Render interpolated frames
        int frameIndex = 0;
        tsd::rendering::Manipulator manipulator;

        for (const auto &pose : samplesCopy) {
          if (m_cancelRequested) {
            tsd::core::logInfo(
                "[CameraPoses] Rendering cancelled at frame %d/%d",
                frameIndex,
                capturedTotalFrames);
            break;
          }

          m_currentFrame = frameIndex;
          manipulator.setConfig(pose);
          tsd::rendering::updateCameraParametersPerspective(d, c, manipulator);
          anari::commitParameters(d, c);

          // Update viewport camera if requested
          if (updateViewport) {
            std::lock_guard<std::mutex> lock(m_poseMutex);
            m_currentPose = pose;
            m_hasNewPose.store(true);
          }

          // Setup output filename with prefix
          std::ostringstream ss;
          if (!capturedFilePrefix.empty()) {
            ss << capturedFilePrefix << "_";
          }
          ss << std::setfill('0') << std::setw(4) << frameIndex << ".png";
          std::filesystem::path filename =
              std::filesystem::path(capturedOutputDirectory) / ss.str();
          savePass->setFilename(filename.string());

          for (int sampleIdx = 0; sampleIdx < config.frame.samples;
              ++sampleIdx) {
            savePass->setEnabled(sampleIdx == config.frame.samples - 1);
            pipeline->render();
          }
          frameIndex++;
        }

        // Cleanup
        pipeline.reset();
        core->anari.releaseRenderIndex(d);
        anari::release(d, c);
        anari::release(d, r);
        anari::release(d, d);
      });
}

} // namespace tsd::ui::imgui
