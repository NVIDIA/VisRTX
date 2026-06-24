// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ShotEditor.h"

#include "imgui.h"
#include "tsd/app/Context.h"
#include "tsd/core/Logging.hpp"
#include "tsd/scene/objects/Renderer.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace tsd::scivis_studio {

namespace {

constexpr const char *NO_RENDERERS_LABEL = "<no renderers>";

std::string rendererLabel(const tsd::scene::Renderer &renderer)
{
  std::string label = renderer.name();
  if (label.empty())
    label = renderer.subtype().str();
  label += " [" + std::to_string(renderer.index()) + "]";
  return label;
}

} // namespace

ShotEditor::ShotEditor(tsd::ui::imgui::Application *app,
    ProjectContext *projectContext,
    std::function<void()> onRender)
    : Window(app, "Shot Editor"),
      m_projectContext(projectContext),
      m_onRender(std::move(onRender))
{}

ShotEditor::~ShotEditor() = default;

bool ShotEditor::inputText(
    const char *label, std::string &value, size_t capacity)
{
  std::vector<char> buffer(capacity, '\0');
  std::strncpy(buffer.data(), value.c_str(), buffer.size() - 1);
  if (ImGui::InputText(label, buffer.data(), buffer.size())) {
    value = buffer.data();
    return true;
  }
  return false;
}

void ShotEditor::buildUI_deviceSelector(Shot &shot)
{
  auto *ctx = m_projectContext ? m_projectContext->appContext() : nullptr;
  auto &project = m_projectContext->project();
  auto &settings = shot.renderSettings;
  const auto preview = settings.rendererLibrary.empty()
      ? std::string{"<none>"}
      : settings.rendererLibrary;

  if (!ctx) {
    ImGui::BeginDisabled();
    if (ImGui::BeginCombo("Device", preview.c_str()))
      ImGui::EndCombo();
    ImGui::EndDisabled();
    return;
  }

  if (ImGui::BeginCombo("Device", preview.c_str())) {
    for (const auto &libName : ctx->anari.libraryList()) {
      const bool selected = settings.rendererLibrary == libName;
      if (ImGui::Selectable(libName.c_str(), selected)) {
        if (settings.rendererLibrary != libName) {
          settings.rendererLibrary = libName;
          settings.rendererObjectIndex = TSD_INVALID_INDEX;
          settings.rendererSubtype = "default";
          m_rendererLoadAttemptedLibrary.clear();
          project.markDirty();
        }
      }
      if (selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
}

void ShotEditor::buildUI_rendererSelector(Shot &shot)
{
  auto *ctx = m_projectContext ? m_projectContext->appContext() : nullptr;
  auto &project = m_projectContext->project();
  auto &settings = shot.renderSettings;
  std::vector<tsd::scene::RendererAppRef> renderers;

  if (ctx && ctx->anari.isLoadableLibrary(settings.rendererLibrary)) {
    auto &scene = ctx->tsd.scene;
    renderers = scene.renderersOfDevice(settings.rendererLibrary);
    if (renderers.empty()
        && m_rendererLoadAttemptedLibrary != settings.rendererLibrary) {
      m_rendererLoadAttemptedLibrary = settings.rendererLibrary;
      if (auto device = ctx->anari.loadDevice(settings.rendererLibrary)) {
        renderers =
            scene.createStandardRenderers(settings.rendererLibrary, device);
        anari::release(device, device);
      } else {
        tsd::core::logWarning(
            "[SciVisStudio] failed to load ANARI device '%s' for shot "
            "renderer selection",
            settings.rendererLibrary.c_str());
      }
    }
  }

  tsd::scene::RendererAppRef currentRenderer;
  if (ctx && settings.rendererObjectIndex != TSD_INVALID_INDEX) {
    auto renderer = ctx->tsd.scene.getObject<tsd::scene::Renderer>(
        settings.rendererObjectIndex);
    if (renderer && renderer->rendererDeviceName() == settings.rendererLibrary)
      currentRenderer = renderer;
  }

  if (!currentRenderer && !renderers.empty()) {
    currentRenderer = renderers.front();
    if (settings.rendererObjectIndex != currentRenderer->index()
        || settings.rendererSubtype != currentRenderer->subtype().str()) {
      settings.rendererObjectIndex = currentRenderer->index();
      settings.rendererSubtype = currentRenderer->subtype().str();
      project.markDirty();
    }
  }

  const auto preview = currentRenderer ? rendererLabel(*currentRenderer)
                                       : std::string{NO_RENDERERS_LABEL};
  ImGui::BeginDisabled(renderers.empty());
  if (ImGui::BeginCombo("Renderer", preview.c_str())) {
    for (const auto &renderer : renderers) {
      if (!renderer)
        continue;
      const bool selected = renderer->index() == settings.rendererObjectIndex;
      const auto label = rendererLabel(*renderer);
      if (ImGui::Selectable(label.c_str(), selected)) {
        if (settings.rendererObjectIndex != renderer->index()
            || settings.rendererSubtype != renderer->subtype().str()) {
          settings.rendererObjectIndex = renderer->index();
          settings.rendererSubtype = renderer->subtype().str();
          project.markDirty();
        }
      }
      if (selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
}

void ShotEditor::buildUI_lightRigSelector(Shot &shot)
{
  auto &project = m_projectContext->project();
  std::string preview = "None";
  if (!shot.lightRigId.empty()) {
    if (auto *rig = light_rig::findLightRig(project, shot.lightRigId))
      preview = rig->name;
    else
      preview = "<missing: " + shot.lightRigId + ">";
  }

  if (ImGui::BeginCombo("Light Rig", preview.c_str())) {
    const bool noneSelected = shot.lightRigId.empty();
    if (ImGui::Selectable("None", noneSelected)) {
      if (!shot.lightRigId.empty()) {
        shot.lightRigId.clear();
        project.markDirty();
        m_projectContext->applyActiveShot();
      }
    }
    if (noneSelected)
      ImGui::SetItemDefaultFocus();

    for (const auto &rig : project.lightRigs) {
      const bool selected = shot.lightRigId == rig.id;
      if (ImGui::Selectable(rig.name.c_str(), selected)) {
        if (shot.lightRigId != rig.id) {
          shot.lightRigId = rig.id;
          project.markDirty();
          m_projectContext->applyActiveShot();
        }
      }
      if (selected)
        ImGui::SetItemDefaultFocus();
    }

    if (!shot.lightRigId.empty()
        && !light_rig::findLightRig(project, shot.lightRigId)) {
      const auto missing = "<missing: " + shot.lightRigId + ">";
      ImGui::TextDisabled("%s", missing.c_str());
    }
    ImGui::EndCombo();
  }
}

void ShotEditor::buildUI_cameraRigSelector(Shot &shot)
{
  auto &project = m_projectContext->project();
  std::string preview = "None";
  if (!shot.cameraRigId.empty()) {
    if (auto *rig = camera_rig::findCameraRig(project, shot.cameraRigId))
      preview = rig->name;
    else
      preview = "<missing: " + shot.cameraRigId + ">";
  }

  if (ImGui::BeginCombo("Camera Rig", preview.c_str())) {
    const bool noneSelected = shot.cameraRigId.empty();
    if (ImGui::Selectable("None", noneSelected)) {
      if (!shot.cameraRigId.empty()) {
        shot.cameraRigId.clear();
        project.markDirty();
        m_projectContext->applyActiveShot();
      }
    }
    if (noneSelected)
      ImGui::SetItemDefaultFocus();

    for (const auto &rig : project.cameraRigs) {
      const bool selected = shot.cameraRigId == rig.id;
      if (ImGui::Selectable(rig.name.c_str(), selected)) {
        if (shot.cameraRigId != rig.id) {
          shot.cameraRigId = rig.id;
          project.markDirty();
          m_projectContext->applyActiveShot();
        }
      }
      if (selected)
        ImGui::SetItemDefaultFocus();
    }

    if (!shot.cameraRigId.empty()
        && !camera_rig::findCameraRig(project, shot.cameraRigId)) {
      const auto missing = "<missing: " + shot.cameraRigId + ">";
      ImGui::TextDisabled("%s", missing.c_str());
    }
    ImGui::EndCombo();
  }
}

void ShotEditor::buildUI()
{
  if (!m_projectContext)
    return;

  auto &project = m_projectContext->project();
  auto *shot = project::activeShot(project);
  if (!shot) {
    ImGui::TextDisabled("No active shot");
    return;
  }
  auto *ctx = m_projectContext->appContext();

  if (inputText("Name", shot->name))
    project.markDirty();

  int currentFrame = shot->currentFrame;
  int frameCount = shot->frameCount;
  float fps = shot->fps;
  if (ImGui::InputInt("Current frame", &currentFrame)) {
    shot->currentFrame = std::clamp(currentFrame, 0, shot->frameCount - 1);
    project.markDirty();
    if (ctx)
      ctx->tsd.animationMgr.setAnimationFrame(shot->currentFrame);
    else
      m_projectContext->applyActiveShot();
  }
  bool playbackSettingsChanged = false;
  playbackSettingsChanged |= ImGui::InputInt("Frame count", &frameCount);
  playbackSettingsChanged |= ImGui::InputFloat("FPS", &fps);
  if (playbackSettingsChanged) {
    shot->frameCount = std::max(1, frameCount);
    shot->currentFrame =
        std::clamp(shot->currentFrame, 0, shot->frameCount - 1);
    shot->fps = std::max(1.f, fps);
    project.markDirty();
    m_projectContext->syncAnimationManagerToActiveShot();
    m_projectContext->applyActiveShot();
  }

  const bool playing = ctx ? ctx->tsd.animationMgr.isPlaying() : shot->playing;
  if (ImGui::Button(playing ? "Stop" : "Play")) {
    if (ctx) {
      if (ctx->tsd.animationMgr.isPlaying())
        ctx->tsd.animationMgr.stop();
      else
        ctx->tsd.animationMgr.play();
      shot->playing = ctx->tsd.animationMgr.isPlaying();
    } else
      shot->playing = !shot->playing;
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("Loop", &shot->loop)) {
    project.markDirty();
    m_projectContext->syncAnimationManagerToActiveShot();
  }

  ImGui::SeparatorText("Render");
  int width = static_cast<int>(shot->renderSettings.width);
  int height = static_cast<int>(shot->renderSettings.height);
  int samples = static_cast<int>(shot->renderSettings.samples);
  if (ImGui::InputInt("Width", &width)) {
    shot->renderSettings.width = static_cast<uint32_t>(std::max(1, width));
    project.markDirty();
  }
  if (ImGui::InputInt("Height", &height)) {
    shot->renderSettings.height = static_cast<uint32_t>(std::max(1, height));
    project.markDirty();
  }
  if (ImGui::InputInt("Samples", &samples)) {
    shot->renderSettings.samples = static_cast<uint32_t>(std::max(1, samples));
    project.markDirty();
  }
  buildUI_deviceSelector(*shot);
  buildUI_rendererSelector(*shot);
  if (inputText("Output prefix", shot->renderSettings.outputFilePrefix))
    project.markDirty();
  buildUI_lightRigSelector(*shot);
  buildUI_cameraRigSelector(*shot);

  ImGui::Text("Output: renders/%s/", shot->id.c_str());
  if (ImGui::Button("Render Active Shot") && m_onRender)
    m_onRender();

  ImGui::SeparatorText("Datasets");
  for (const auto &dataset : project.datasets) {
    bool enabled = true;
    if (auto *binding = shot::findDatasetBinding(*shot, dataset.id))
      enabled = binding->enabled;
    if (ImGui::Checkbox(dataset.name.c_str(), &enabled)) {
      shot::setDatasetBinding(*shot, dataset.id, enabled);
      project.markDirty();
      m_projectContext->applyActiveShot();
    }
  }
}

} // namespace tsd::scivis_studio
