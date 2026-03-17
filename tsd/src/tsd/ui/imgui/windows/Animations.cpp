// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Animations.h"
// tsd_core
#include "tsd/animation/AnimationManager.hpp"
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
  auto &animMgr = ctx->tsd.animationMgr;

  if (m_playing)
    animMgr.incrementAnimationTime();

  buildUI_animationControls();

  size_t toDelete = SIZE_MAX;
  auto &anims = animMgr.animations();
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
    animMgr.removeAnimation(toDelete);
}

void Animations::buildUI_animationControls()
{
  auto *ctx = appContext();
  auto &animMgr = ctx->tsd.animationMgr;

  ImGui::BeginDisabled(m_playing);

  float time = animMgr.getAnimationTime();
  if (ImGui::SliderFloat("time", &time, 0.f, 1.f))
    animMgr.setAnimationTime(time);

  if (ImGui::Button("play"))
    m_playing = true;
  ImGui::EndDisabled();

  ImGui::BeginDisabled(!m_playing);
  ImGui::SameLine();
  if (ImGui::Button("stop"))
    m_playing = false;
  ImGui::EndDisabled();

  ImGui::SameLine();
  float increment = animMgr.getAnimationIncrement();
  if (ImGui::DragFloat("step", &increment, 0.01f, 0.f, 0.5f))
    animMgr.setAnimationIncrement(increment);
}


} // namespace tsd::ui::imgui
