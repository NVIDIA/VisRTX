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

  auto *ctx = appContext();
  auto &scene = ctx->tsd.scene;

  if (m_playing)
    scene.sceneAnimation().incrementAnimationTime();

  buildUI_animationControls();

  size_t toDelete = SIZE_MAX;
  auto &anims = scene.sceneAnimation().animations();
  for (size_t i = 0; i < anims.size(); i++) {
    auto &anim = anims[i];
    ImGui::PushID(static_cast<int>(i));
    ImGui::Separator();
    ImGui::Text("name | %s", anim.name.c_str());
    ImGui::Text("info | %zu bindings, %zu transforms",
        anim.bindings.size(),
        anim.transforms.size());
    if (ImGui::Button("delete"))
      toDelete = i;
    ImGui::PopID();
  }
  if (toDelete != SIZE_MAX)
    scene.sceneAnimation().removeAnimation(toDelete);
}

void Animations::buildUI_animationControls()
{
  auto *ctx = appContext();
  auto &scene = ctx->tsd.scene;

  ImGui::BeginDisabled(m_playing);

  float time = scene.sceneAnimation().getAnimationTime();
  if (ImGui::SliderFloat("time", &time, 0.f, 1.f))
    scene.sceneAnimation().setAnimationTime(time);

  if (ImGui::Button("play"))
    m_playing = true;
  ImGui::EndDisabled();

  ImGui::BeginDisabled(!m_playing);
  ImGui::SameLine();
  if (ImGui::Button("stop"))
    m_playing = false;
  ImGui::EndDisabled();

  ImGui::SameLine();
  float increment = scene.sceneAnimation().getAnimationIncrement();
  if (ImGui::DragFloat("step", &increment, 0.01f, 0.f, 0.5f))
    scene.sceneAnimation().setAnimationIncrement(increment);
}


} // namespace tsd::ui::imgui
