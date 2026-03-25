// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Animations.h"
// tsd_core
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/Logging.hpp"
// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
// std
#include <optional>

namespace tsd::ui::imgui {

Animations::Animations(Application *app, const char *name) : Window(app, name)
{}

void Animations::buildUI()
{
  auto *ctx = appContext();
  auto &animMgr = ctx->tsd.animationMgr;

  buildUI_animationControls();

  auto &anims = animMgr.animations();
  if (anims.empty()) {
    ImGui::Separator();
    ImGui::Text("-- No Animations --");
  } else {
    std::optional<size_t> toDelete;
    for (size_t i = 0; i < anims.size(); i++) {
      auto &anim = anims[i];
      ImGui::PushID(static_cast<int>(i));
      ImGui::Separator();
      ImGui::Text("name | %s", anim.name().c_str());
      ImGui::Text("info | %zu parameter, %zu transforms, %zu callbacks",
          anim.objectParameterBindings().size(),
          anim.transformBindings().size(),
          anim.callbackBindings().size());
      if (ImGui::Button("delete"))
        toDelete = i;
      ImGui::PopID();
    }
    if (toDelete)
      animMgr.removeAnimation(toDelete.value());
  }
}

void Animations::buildUI_animationControls()
{
  auto *ctx = appContext();
  auto &animMgr = ctx->tsd.animationMgr;

  ImGui::BeginDisabled(animMgr.isPlaying());

  float time = animMgr.getAnimationTime();
  if (ImGui::SliderFloat("time", &time, 0.f, 1.f))
    animMgr.setAnimationTime(time);

  if (ImGui::Button("play"))
    animMgr.play();
  ImGui::EndDisabled();

  ImGui::BeginDisabled(!animMgr.isPlaying());
  ImGui::SameLine();
  if (ImGui::Button("stop"))
    animMgr.stop();
  ImGui::EndDisabled();

  ImGui::SameLine();
  float increment = animMgr.getAnimationIncrement();
  if (ImGui::DragFloat("step", &increment, 0.01f, 0.f, 0.5f))
    animMgr.setAnimationIncrement(increment);
}

} // namespace tsd::ui::imgui
