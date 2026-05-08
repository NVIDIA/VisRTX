// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ShotEditor.h"

#include "imgui.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace tsd::scivis_studio {

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

void ShotEditor::buildUI()
{
  if (!m_projectContext)
    return;

  auto &project = m_projectContext->project();
  auto *shot = activeShot(project);
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
  if (inputText("Renderer library", shot->renderSettings.rendererLibrary))
    project.markDirty();
  if (inputText("Renderer subtype", shot->renderSettings.rendererSubtype))
    project.markDirty();
  if (inputText("Output prefix", shot->renderSettings.outputFilePrefix))
    project.markDirty();

  ImGui::Text("Output: renders/%s/", shot->id.c_str());
  if (ImGui::Button("Render Active Shot") && m_onRender)
    m_onRender();

  ImGui::SeparatorText("Datasets");
  for (const auto &dataset : project.datasets) {
    bool enabled = true;
    if (auto *binding = findDatasetBinding(*shot, dataset.id))
      enabled = binding->enabled;
    if (ImGui::Checkbox(dataset.name.c_str(), &enabled)) {
      setDatasetBinding(*shot, dataset.id, enabled);
      project.markDirty();
      m_projectContext->applyActiveShot();
    }
  }
}

} // namespace tsd::scivis_studio
