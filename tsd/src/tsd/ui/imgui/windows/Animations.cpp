// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Animations.h"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"

namespace tsd::ui::imgui {

Animations::Animations(Application *app, const char *name) : Window(app, name)
{}

void Animations::buildUI()
{
  if (ImGui::IsKeyPressed(ImGuiKey_Space))
    m_playing = !m_playing;

  auto *core = appCore();
  auto &scene = core->tsd.scene;

  if (m_playing)
    scene.incrementAnimationTime();

  buildUI_animationControls();

  tsd::core::Animation *toDelete = nullptr;
  for (size_t i = 0; i < scene.numberOfAnimations(); i++) {
    auto *animation = scene.animation(i);
    buildUI_editAnimation(animation);
    if (ImGui::Button("delete"))
      toDelete = animation;
  }
  if (toDelete != nullptr)
    scene.removeAnimation(toDelete);
}

void Animations::buildUI_animationControls()
{
  auto *core = appCore();
  auto &scene = core->tsd.scene;

  ImGui::BeginDisabled(m_playing);

  float time = scene.getAnimationTime();
  if (ImGui::SliderFloat("time", &time, 0.f, 1.f))
    scene.setAnimationTime(time);

  if (ImGui::Button("play"))
    m_playing = true;
  ImGui::EndDisabled();

  ImGui::BeginDisabled(!m_playing);
  ImGui::SameLine();
  if (ImGui::Button("stop"))
    m_playing = false;
  ImGui::EndDisabled();

  ImGui::SameLine();
  float increment = scene.getAnimationIncrement();
  if (ImGui::DragFloat("step", &increment, 0.01f, 0.f, 0.5f))
    scene.setAnimationIncrement(increment);
}

void Animations::buildUI_editAnimation(tsd::core::Animation *animation)
{
  ImGui::Separator();
  ImGui::Text("name | %s", animation->name().c_str());
  ImGui::Text("info | %s", animation->info().c_str());
}

} // namespace tsd::ui::imgui
